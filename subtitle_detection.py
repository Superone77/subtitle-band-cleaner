"""Detect changing bottom captions and estimate their processing band."""

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from math import ceil
import re
from typing import Optional


@dataclass
class BandEstimate:
    width: int
    height: int
    duration_seconds: float
    sampled_frames: int
    subtitle_frames: int
    top_pixel: Optional[int]
    bottom_percent: Optional[float]
    has_subtitles: bool = True
    candidate_frames: int = 0
    distinct_captions: int = 0
    reason: str = "changing_bottom_captions"

    def as_dict(self):
        return asdict(self)


@dataclass
class CaptionObservation:
    time: float
    top: float
    bottom: float
    left: float
    right: float
    text: str


def normalized_text(text):
    return re.sub(r"[^\w]", "", text.casefold())


def distinct_text_count(observations):
    """Collapse small OCR spelling changes of the same static label."""
    representatives = []
    for observation in observations:
        text = normalized_text(observation.text)
        if not text:
            continue
        if not any(SequenceMatcher(None, text, other).ratio() >= 0.78
                   for other in representatives):
            representatives.append(text)
    return len(representatives)


def changing_caption_tracks(observations, width, height, min_distinct=3):
    """Require changing text at a consistent baseline and alignment."""
    from statistics import median

    tracks = []
    for observation in observations:
        center = (observation.left + observation.right) / 2
        for track in tracks:
            baseline = median(item.bottom for item in track)
            track_center = median((item.left + item.right) / 2 for item in track)
            if abs(observation.bottom - baseline) <= 0.035 * height and abs(center - track_center) <= 0.12 * width:
                track.append(observation)
                break
        else:
            tracks.append([observation])
    counts = [distinct_text_count(track) for track in tracks]
    accepted = [track for track, count in zip(tracks, counts)
                if count >= min_distinct]
    return accepted, max(counts, default=0)


def detect_bottom_band(path, sample_interval=1.0, max_samples=600,
                       min_confidence=0.80, margin_pixels=8,
                       min_distinct_captions=3, alignment="center"):
    """Return a classification, including a normal no-subtitle result.

    Scene text and static nameplates are not sufficient subtitle evidence.
    Decode failures remain errors and never become no-subtitle decisions.
    """
    import cv2
    import numpy as np
    from rapidocr import RapidOCR

    if sample_interval <= 0 or max_samples < 1 or margin_pixels < 0 or min_distinct_captions < 1:
        raise ValueError("Invalid detection parameters")
    if alignment not in {"center", "any"}:
        raise ValueError("Alignment must be center or any")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Cannot decode video: " + str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError("Cannot read video timing or dimensions: " + str(path))
        duration = frame_count / fps
        sample_count = min(max_samples, max(1, ceil(duration / sample_interval)))
        # Target actual frame indices, including the final decodable frame.
        indices = np.linspace(0, int(frame_count) - 1, min(sample_count, int(frame_count)), dtype=int)
        ocr = RapidOCR()
        observations = []
        sampled = 0
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                continue
            sampled += 1
            result = ocr(frame)
            if result.boxes is None:
                continue
            candidates = []
            for box, score, text in zip(result.boxes, result.scores, result.txts):
                if score < min_confidence or len(normalized_text(text)) < 2:
                    continue
                left = float(np.min(box[:, 0]))
                right = float(np.max(box[:, 0]))
                top = float(np.min(box[:, 1]))
                bottom = float(np.max(box[:, 1]))
                center_x = (left + right) / 2
                line_height = bottom - top
                low, high = (0.25, 0.75) if alignment == "center" else (0.05, 0.95)
                if not (low * width <= center_x <= high * width):
                    continue
                if top < 0.65 * height or bottom > height:
                    continue
                if not (0.01 * height <= line_height <= 0.16 * height):
                    continue
                if right - left < 0.035 * width:
                    continue
                candidates.append(CaptionObservation(index / fps, top, bottom, left, right, text))
            anchors = [item for item in candidates if item.bottom >= 0.84 * height]
            if not anchors:
                continue
            anchor = max(anchors, key=lambda item: item.bottom)
            center = (anchor.left + anchor.right) / 2
            lines = [item for item in candidates
                     if abs((item.left + item.right) / 2 - center) <= 0.12 * width
                     and item.bottom <= anchor.bottom
                     and item.top >= anchor.top - 0.14 * height]
            lines.sort(key=lambda item: (item.top, item.left))
            observations.append(CaptionObservation(
                index / fps, min(item.top for item in lines), anchor.bottom,
                anchor.left, anchor.right, " ".join(item.text for item in lines)))
        if sampled < max(1, ceil(len(indices) * 0.8)):
            raise ValueError("Too many sampled frames could not be decoded: " + str(path))
        tracks, distinct = changing_caption_tracks(observations, width, height, min_distinct_captions)
        captions = [item for track in tracks for item in track]
        if not captions:
            return BandEstimate(
                width, height, duration, sampled, 0, None, None, False,
                len(observations), distinct,
                "no_bottom_text" if not observations else "no_changing_caption_track")
        top = max(0, int(np.percentile([item.top for item in captions], 5)) - margin_pixels)
        top -= top % 2
        percent = 100 * (height - top) / height
        if percent >= 60:
            raise ValueError("Detected band is implausibly large for " + str(path))
        return BandEstimate(width, height, duration, sampled, len(captions), top,
                            percent, True, len(observations), distinct)
    finally:
        capture.release()
