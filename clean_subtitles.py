#!/usr/bin/env python3
"""Automatically find and crop or blur burned-in bottom subtitles."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
        if is_processed(path):
            raise ValueError("Input is already a processed output: " + str(path))
        return [path]
    if path.is_dir():
        files = sorted(
            item for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES
            and not item.name.startswith(".") and not is_processed(item)
        )
        if not files:
            raise ValueError("No MP4, MOV, or MKV videos in " + str(path))
        return files
    raise ValueError("Input does not exist: " + str(path))


def is_processed(path):
    return path.stem.endswith(("_subtitle_blurred", "_subtitle_cropped"))


def base_stem(path):
    stem = path.stem
    while stem.endswith(("_no_subtitle", "_with_subtitle")):
        stem = stem.rsplit("_", 2)[0]
    return stem


def rename_original(source, classified):
    if source == classified:
        return
    if classified.exists():
        raise ValueError("Will not overwrite another original: " + str(classified))
    source.rename(classified)


def process_video(source, args):
    estimate = detect_bottom_band(
        source, args.sample_interval, args.max_samples,
        margin_pixels=args.margin_pixels,
        min_distinct_captions=args.min_distinct_captions,
        alignment=args.alignment)
    has_subtitles = estimate.has_subtitles or args.force_process
    tag = "with_subtitle" if has_subtitles else "no_subtitle"
    stem = base_stem(source)
    classified = source.with_name(stem + "_" + tag + source.suffix)
    original = classified if args.rename_originals else source
    if original != source and original.exists():
        raise ValueError("Will not overwrite another original: " + str(original))
    percent = args.bottom_percent if args.bottom_percent is not None else estimate.bottom_percent
    target = None
    if has_subtitles and not args.classify_only:
        suffix = "subtitle_cropped" if args.mode == "crop" else "subtitle_blurred"
        target = args.output / (stem + "_" + suffix + ".mp4")
        if target.exists() and not args.overwrite:
            raise ValueError("Output already exists (use --overwrite): " + str(target))
        ffmpeg = ffmpeg_executable(args.ffmpeg)
        with tempfile.NamedTemporaryFile(dir=args.output, prefix="." + stem + ".",
                                         suffix=".partial.mp4", delete=False) as file:
            temporary = Path(file.name)
        print("{}: subtitles detected; {} bottom {:.1f}% -> {}".format(
            source.name, args.mode, percent, target), flush=True)
        try:
            subprocess.run(make_command(ffmpeg, source, temporary, args.mode,
                                        percent, args.crf, args.preset), check=True)
            rename_original(source, original)
            os.replace(str(temporary), str(target))
        finally:
            temporary.unlink(missing_ok=True)
    elif args.rename_originals:
        rename_original(source, original)
        print("{}: {} -> {} (video bytes unchanged)".format(
            source.name, tag, original), flush=True)
    else:
        print("{}: {} (input unchanged)".format(source.name, tag), flush=True)
    report = {
        "input": source.name,
        "classified_input": str(original),
        "classification": tag,
        "renamed": original != source,
        "has_subtitles": has_subtitles,
        "processed": target is not None,
        "forced": args.force_process,
        "output": target.name if target else None,
        "mode": args.mode if target else None,
        "bottom_percent": percent if has_subtitles else None,
        "detection": estimate.as_dict(),
    }
    report_path = target.with_suffix(".json") if target else args.output / (stem + "_" + tag + ".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


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
                        help="override band height after detection; does not force processing")
    parser.add_argument("--classify-only", action="store_true",
                        help="detect and write reports without encoding outputs")
    parser.add_argument("--rename-originals", action="store_true",
                        help="rename inputs to *_with_subtitle / *_no_subtitle")
    parser.add_argument("--force-process", action="store_true",
                        help="process even without OCR evidence; requires --bottom-percent")
    parser.add_argument("--min-distinct-captions", type=int, default=3,
                        help="minimum different captions at a consistent position (default: 3)")
    parser.add_argument("--alignment", choices=("center", "any"), default="center",
                        help="candidate caption alignment (default: center)")
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
        if args.min_distinct_captions < 1:
            raise ValueError("Minimum distinct captions must be positive")
        if args.force_process and args.bottom_percent is None:
            raise ValueError("--force-process requires --bottom-percent")
        sources = inputs_at(args.input)
        args.output.mkdir(parents=True, exist_ok=True)
        failed = 0
        for source in sources:
            try:
                process_video(source, args)
            except (ValueError, OSError, subprocess.CalledProcessError) as exc:
                failed += 1
                print("Error: {}: {}".format(source.name, exc), file=sys.stderr)
        return 1 if failed else 0
    except (ValueError, OSError) as exc:
        print("Error: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
