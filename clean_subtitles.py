#!/usr/bin/env python3
"""Crop or strongly blur a configurable band at the bottom of videos."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from subtitle_detection import detect_bottom_band


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv"}


def percentage(value):
    number = float(value)
    if not 0 < number < 60:
        raise argparse.ArgumentTypeError("bottom percentage must be between 0 and 60")
    return number


def ffmpeg_executable(explicit):
    if explicit:
        path = shutil.which(explicit) or explicit
        if not Path(path).is_file():
            raise ValueError("FFmpeg executable not found: " + explicit)
        return path
    installed = shutil.which("ffmpeg")
    if installed:
        return installed
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise ValueError("Install FFmpeg or run: pip install -r requirements.txt") from exc


def inputs_at(path):
    if path.is_file():
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError("Input must be an MP4, MOV, or MKV video")
        return [path]
    if path.is_dir():
        files = sorted(
            item for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES
        )
        if not files:
            raise ValueError("No MP4, MOV, or MKV videos in " + str(path))
        return files
    raise ValueError("Input does not exist: " + str(path))


def video_filter(mode, percent):
    fraction = percent / 100
    # Chroma-subsampled H.264 needs an even output height.
    band = "trunc(ih*{:.6f}/2)*2".format(fraction)
    if mode == "crop":
        kept = "trunc(ih*(1-{:.6f})/2)*2".format(fraction)
        return "-vf", "crop=iw:{}:0:0".format(kept), "0:v:0"
    graph = (
        "[0:v:0]split=2[base][source];"
        "[source]crop=iw:{}:0:ih-oh,gblur=sigma=18[band];"
        "[base][band]overlay=0:H-h:shortest=1[v]"
    ).format(band)
    return "-filter_complex", graph, "[v]"


def make_command(ffmpeg, source, destination, mode, percent, crf, preset):
    filter_flag, filter_value, video_map = video_filter(mode, percent)
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-stats", "-y",
        "-i", str(source), filter_flag, filter_value,
        "-map", video_map, "-map", "0:a?", "-sn", "-dn",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        "-movflags", "+faststart", str(destination),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="one video or a directory of videos")
    parser.add_argument("output", type=Path, help="output directory")
    parser.add_argument("--mode", choices=("crop", "blur"), default="crop")
    parser.add_argument("--bottom-percent", type=percentage,
                        help="manual override; by default OCR detects each video's band")
    parser.add_argument("--sample-interval", type=float, default=1.0,
                        help="seconds between OCR samples (default: 1)")
    parser.add_argument("--max-samples", type=int, default=600,
                        help="maximum OCR samples per video (default: 600)")
    parser.add_argument("--margin-pixels", type=int, default=8,
                        help="extra pixels above detected text (default: 8)")
    parser.add_argument("--ffmpeg", help="path to FFmpeg executable")
    parser.add_argument("--crf", type=int, default=18,
                        help="H.264 quality (lower is better; default: 18)")
    parser.add_argument("--preset", default="medium",
                        choices=("ultrafast", "superfast", "veryfast", "faster",
                                 "fast", "medium", "slow", "slower", "veryslow"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not 0 <= args.crf <= 51:
            raise ValueError("CRF must be between 0 and 51")
        if args.sample_interval <= 0 or args.max_samples < 1 or args.margin_pixels < 0:
            raise ValueError("Sample interval and max samples must be positive; margin cannot be negative")
        sources = inputs_at(args.input)
        ffmpeg = ffmpeg_executable(args.ffmpeg)
        args.output.mkdir(parents=True, exist_ok=True)
        for source in sources:
            estimate = None
            if args.bottom_percent is None:
                estimate = detect_bottom_band(source, args.sample_interval,
                                              args.max_samples,
                                              margin_pixels=args.margin_pixels)
                percent = estimate.bottom_percent
            else:
                percent = args.bottom_percent
            target = args.output / (source.stem + "_" + args.mode + ".mp4")
            if source.resolve() == target.resolve():
                raise ValueError("Output would replace input: " + str(source))
            if target.exists() and not args.overwrite:
                raise ValueError("Output already exists (use --overwrite): " + str(target))
            temporary = target.with_name("." + target.stem + ".partial.mp4")
            print("{}: {} bottom {:.1f}% -> {}".format(source.name, args.mode,
                                                       percent, target), flush=True)
            try:
                command = make_command(ffmpeg, source, temporary, args.mode,
                                       percent, args.crf, args.preset)
                subprocess.run(command, check=True)
                os.replace(str(temporary), str(target))
                report = {
                    "input": source.name,
                    "output": target.name,
                    "mode": args.mode,
                    "bottom_percent": percent,
                    "detection": estimate.as_dict() if estimate else "manual",
                }
                target.with_suffix(".json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
            finally:
                temporary.unlink(missing_ok=True)
    except subprocess.CalledProcessError as exc:
        print("Error: FFmpeg failed with exit code {}".format(exc.returncode), file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print("Error: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
