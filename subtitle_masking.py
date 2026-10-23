"""Mask detected caption text boxes only in frames where captions are shown."""

from contextlib import suppress
from fractions import Fraction
from math import ceil, floor
from pathlib import Path
import shutil
import subprocess

from subtitle_detection import CaptionObservation, changing_caption_tracks, normalized_text


FILLS = ("blur", "pixelate", "inpaint")
# Captions touching the bottom edge are often missed by the text detector;
# a black strip below the band crop recovers them (boxes are clipped back).
OCR_PADDING = 32


def caption_lines(result, crop_top, width, height, tracks, time=0.0,
                  min_confidence=0.80, alignment="center"):
    """Return one sample's caption line boxes and merged observation.

    Uses detection's candidate rules. With no tracks (forced processing),
    every candidate in the band counts as a caption line.
    """
    if result.boxes is None:
        return [], None
    low, high = (0.25, 0.75) if alignment == "center" else (0.05, 0.95)
    candidates = []
    for box, score, text in zip(result.boxes, result.scores, result.txts):
        if score < min_confidence or len(normalized_text(text)) < 2:
            continue
        left = float(min(point[0] for point in box))
        right = float(max(point[0] for point in box))
        top = float(min(point[1] for point in box)) + crop_top
        bottom = min(float(max(point[1] for point in box)) + crop_top, height)
        if not (low * width <= (left + right) / 2 <= high * width):
            continue
        # Detection caps lines at 16% of the height; boxes that must also sit on
        # a confirmed track may be taller (large fonts, descenders), so allow 20%.
        if not (0.01 * height <= bottom - top <= 0.20 * height):
            continue
        if right - left < 0.035 * width:
            continue
        candidates.append(CaptionObservation(time, top, bottom, left, right, text))
    if not candidates:
        return [], None
    if tracks:
        anchors = [item for item in candidates if any(
            abs(item.bottom - track["baseline"]) <= 0.035 * height
            and abs((item.left + item.right) / 2 - track["center_x"]) <= 0.12 * width
            for track in tracks)]
        if not anchors:
            return [], None
        anchor = max(anchors, key=lambda item: item.bottom)
        center = (anchor.left + anchor.right) / 2
        # Unlike detection, accept lines below the anchor too: top-anchored
        # captions grow downwards when a second line is added.
        lines = [item for item in candidates
                 if abs((item.left + item.right) / 2 - center) <= 0.12 * width
                 and item.top >= anchor.top - 0.14 * height
                 and item.bottom <= anchor.bottom + 0.14 * height]
    else:
        anchor = max(candidates, key=lambda item: item.bottom)
        lines = candidates
    lines.sort(key=lambda item: (item.top, item.left))
    boxes = [(floor(item.left), floor(item.top), ceil(item.right), ceil(item.bottom))
             for item in lines]
    # The merged observation keeps the anchor's bottom so it stays on the track.
    merged = CaptionObservation(time, min(item.top for item in lines), anchor.bottom,
                                anchor.left, anchor.right, " ".join(item.text for item in lines))
    return boxes, merged


def caption_samples(samples, width, height, min_distinct=3):
    """Keep boxes only for samples on a changing caption track; drops static text.

    Consecutive samples showing the same text at the same position count as
    one observation, so the track logic does not pay for every repeated sample.
    """
    runs = []
    for index, (_, observation) in enumerate(samples):
        if observation is None:
            continue
        if runs and runs[-1][1][-1] == index - 1 and same_caption(runs[-1][0], observation, width, height):
            runs[-1][1].append(index)
        else:
            runs.append((observation, [index]))
    tracks, _ = changing_caption_tracks([observation for observation, _ in runs],
                                        width, height, min_distinct)
    # Identity, not equality: equal dataclasses from different runs must not match.
    accepted = {id(item) for track in tracks for item in track}
    kept = {index for observation, indices in runs if id(observation) in accepted
            for index in indices}
    return [boxes if index in kept else [] for index, (boxes, _) in enumerate(samples)]


def same_caption(first, second, width, height):
    """Same text on the same baseline and alignment, as the track logic defines it."""
    return (normalized_text(first.text) == normalized_text(second.text)
            and abs(first.bottom - second.bottom) <= 0.035 * height
            and abs((first.left + first.right) - (second.left + second.right)) / 2 <= 0.12 * width)


def mask_timeline(sample_boxes, step, frame_count, hold_samples=3):
    """Return the box list for every frame from boxes sampled every step frames.

    An empty sample holds the latest non-empty one within hold_samples, and
    each frame also gets the next sample's boxes so new text is covered early.
    """
    if not sample_boxes:
        return [[] for _ in range(frame_count)]
    active = []
    for index, boxes in enumerate(sample_boxes):
        if not boxes:
            for back in range(index - 1, max(-1, index - hold_samples - 1), -1):
                if sample_boxes[back]:
                    boxes = sample_boxes[back]
                    break
        active.append(boxes)
    combined = [list(dict.fromkeys(boxes + (sample_boxes[index + 1] if index + 1 < len(sample_boxes) else [])))
                for index, boxes in enumerate(active)]
    last = len(combined) - 1
    return [combined[min(frame // step, last)] for frame in range(frame_count)]


def fill_boxes(frame, boxes, margin_pixels=8, fill="blur"):
    """Return a copy with only the dilated box pixels blurred, pixelated or inpainted."""
    import cv2
    import numpy as np

    if not boxes:
        return frame
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    for x0, y0, x1, y1 in boxes:
        mask[max(0, y0 - margin_pixels):min(height, y1 + margin_pixels),
             max(0, x0 - margin_pixels):min(width, x1 + margin_pixels)] = 255
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        return frame
    tallest = max(y1 - y0 for _, y0, _, y1 in boxes)
    if fill == "blur":
        sigma = max(6, round(0.35 * tallest))
        pad = 3 * sigma
    elif fill == "pixelate":
        block = max(6, round(0.3 * tallest))
        pad = 0
    elif fill == "inpaint":
        pad = 16
    else:
        raise ValueError("Fill must be one of: " + ", ".join(FILLS))
    top, bottom = max(0, rows[0] - pad), min(height, rows[-1] + 1 + pad)
    left, right = max(0, columns[0] - pad), min(width, columns[-1] + 1 + pad)
    roi = frame[top:bottom, left:right]
    mask_roi = mask[top:bottom, left:right]
    if fill == "blur":
        patch = cv2.GaussianBlur(roi, (0, 0), sigma)
    elif fill == "pixelate":
        small = cv2.resize(roi, (max(1, round(roi.shape[1] / block)), max(1, round(roi.shape[0] / block))),
                           interpolation=cv2.INTER_LINEAR)
        patch = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        patch = cv2.inpaint(np.ascontiguousarray(roi), mask_roi, 5, cv2.INPAINT_TELEA)
    output = frame.copy()
    np.copyto(output[top:bottom, left:right], patch, where=mask_roi[:, :, None] > 0)
    return output


def source_timing(ffmpeg, source):
    """Return the video stream's start time and sample aspect ratio via ffprobe.

    Raw frames carry neither, so they are re-applied on encoding. Without
    ffprobe (imageio-ffmpeg ships only ffmpeg) both fall back to defaults.
    """
    ffprobe = shutil.which("ffprobe") or Path(ffmpeg).with_name("ffprobe")
    if not Path(ffprobe).is_file():
        return 0.0, None
    try:
        probe = subprocess.run(
            [str(ffprobe), "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=start_time,sample_aspect_ratio", "-of", "default=nw=1", str(source)],
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0.0, None
    values = dict(line.split("=", 1) for line in probe.splitlines() if "=" in line)
    try:
        start = max(0.0, float(values.get("start_time", 0)))
    except ValueError:
        start = 0.0
    sar = values.get("sample_aspect_ratio", "")
    ratio = sar.split(":") if ":" in sar else None
    if not ratio or not all(part.isdigit() for part in ratio) or "0" in ratio or ratio[0] == ratio[1]:
        sar = None
    return start, sar


def mask_command(ffmpeg, source, destination, width, height, fps, crf, preset, start=0.0, sar=None):
    """Encode raw frames from stdin, keeping the source's audio, start time and pixel shape."""
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-stats", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", "{}x{}".format(width, height),
        "-framerate", str(Fraction(fps).limit_denominator(10000)),
        "-itsoffset", "{:.6f}".format(start), "-i", "-", "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?", "-sn", "-dn",
    ]
    if sar:
        command += ["-vf", "setsar=" + sar.replace(":", "/")]
    return command + [
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        "-movflags", "+faststart", str(destination),
    ]


def mask_video(source, destination, ffmpeg, crf, preset, band_top, caption_tracks,
               sample_fps=5.0, hold_samples=3, margin_pixels=8, fill="blur",
               min_confidence=0.80, min_distinct_captions=3, alignment="center", ocr=None):
    """OCR the band, then re-encode with caption boxes masked frame by frame.

    Frames without caption boxes pass through unchanged before encoding.
    """
    import cv2

    if sample_fps <= 0 or hold_samples < 0 or margin_pixels < 0 or min_distinct_captions < 1:
        raise ValueError("Invalid masking parameters")
    if fill not in FILLS:
        raise ValueError("Fill must be one of: " + ", ".join(FILLS))
    if alignment not in {"center", "any"}:
        raise ValueError("Alignment must be center or any")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Cannot decode video: " + str(source))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError("Cannot read video timing or dimensions: " + str(source))
        if not 0 <= band_top < height:
            raise ValueError("Band top is outside the frame: " + str(band_top))
        step = max(1, round(fps / sample_fps))
        crop_top = max(0, band_top - margin_pixels)
        if ocr is None:
            from rapidocr import RapidOCR

            # The default min side limit upscales a short band crop and is much slower;
            # empty band crops are normal here, so skip RapidOCR's per-frame warning.
            ocr = RapidOCR(params={"Det.limit_type": "max", "Det.limit_side_len": 960,
                                   "Global.log_level": "error"})
        print("{}: mask pass 1/2, OCR every {} frames below y={}".format(
            source.name, step, crop_top), flush=True)
        samples = []
        frames = 0
        while capture.grab():
            if frames % step == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    raise ValueError("Cannot decode frame {} of {}".format(frames, source))
                band = cv2.copyMakeBorder(frame[crop_top:, :], 0, OCR_PADDING, 0, 0,
                                          cv2.BORDER_CONSTANT, value=0)
                samples.append(caption_lines(
                    ocr(band), crop_top, width, height, caption_tracks,
                    frames / fps, min_confidence, alignment))
            frames += 1
    finally:
        capture.release()
    if frames < max(1, ceil(frame_count * 0.8)):
        raise ValueError("Too many frames could not be decoded: " + str(source))
    if caption_tracks:
        sample_boxes = caption_samples(samples, width, height, min_distinct_captions)
    else:
        sample_boxes = [boxes for boxes, _ in samples]
    timeline = mask_timeline(sample_boxes, step, frames, hold_samples)
    report = {
        "fill": fill,
        "sample_fps": sample_fps,
        "hold_samples": hold_samples,
        "samples": len(samples),
        "samples_with_captions": sum(1 for boxes in sample_boxes if boxes),
        "frames": 0,
        "masked_frames": 0,
    }
    print("{}: mask pass 2/2, {} on {} of {} frames ({} of {} samples with captions)".format(
        source.name, fill, sum(1 for boxes in timeline if boxes), frames,
        report["samples_with_captions"], report["samples"]), flush=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Cannot decode video: " + str(source))
    start, sar = source_timing(ffmpeg, source)
    command = mask_command(ffmpeg, source, destination, width, height, fps, crf, preset, start, sar)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if report["frames"] >= frames:
                raise ValueError("Second decode pass returned more frames than the first: " + str(source))
            if frame.shape != (height, width, 3):
                raise ValueError("Frame size changed while decoding: " + str(source))
            boxes = timeline[report["frames"]]
            if boxes:
                frame = fill_boxes(frame, boxes, margin_pixels, fill)
            try:
                process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break
            report["frames"] += 1
            report["masked_frames"] += bool(boxes)
        with suppress(BrokenPipeError):
            process.stdin.close()
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        capture.release()
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    if report["frames"] != frames:
        raise ValueError("Second decode pass returned {} of {} frames: {}".format(
            report["frames"], frames, source))
    return report
