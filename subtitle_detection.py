"""Estimate the subtitle band of a video from OCR samples."""

from dataclasses import asdict, dataclass
from math import ceil


@dataclass
class BandEstimate:
    width: int
    height: int
    duration_seconds: float
    sampled_frames: int
    subtitle_frames: int
    top_pixel: int
    bottom_percent: float

    def as_dict(self):
        return asdict(self)


def detect_bottom_band(path, sample_interval=1.0, max_samples=600,
                       min_confidence=0.80, margin_pixels=8):
    """Find a full-width bottom band that covers detected subtitles.

    OCR boxes are filtered by position and confidence. A low percentile of
    the top edge covers multi-line captions while rejecting isolated high
    false positives. The result is a video-wide band, so it also covers
    subtitle changes between sampled frames.
    """
    import cv2
    import numpy as np
    from rapidocr import RapidOCR

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
        times = np.linspace(0, max(0, duration - 0.05), sample_count)
        ocr = RapidOCR()
        caption_tops = []
        sampled = 0
        for time in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(time) * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            sampled += 1
            result = ocr(frame)
            if result.boxes is None:
                continue
            candidates = []
            for box, score, text in zip(result.boxes, result.scores, result.txts):
                if score < min_confidence or len(text.strip()) < 2:
                    continue
                left = float(np.min(box[:, 0]))
                right = float(np.max(box[:, 0]))
                top = float(np.min(box[:, 1]))
                bottom = float(np.max(box[:, 1]))
                center_x = (left + right) / 2
                line_height = bottom - top
                if not (0.15 * width <= center_x <= 0.85 * width):
                    continue
                if top < 0.65 * height or bottom > height:
                    continue
                if not (0.01 * height <= line_height <= 0.16 * height):
                    continue
                if right - left < 0.035 * width:
                    continue
                candidates.append((top, bottom))
            # Require a line near the bottom; include any preceding lines.
            if candidates and max(bottom for _, bottom in candidates) >= 0.84 * height:
                caption_tops.append(min(top for top, _ in candidates))
        if not caption_tops:
            raise ValueError(
                "No bottom subtitles detected in {}. Check the video or use "
                "--bottom-percent for a manual band.".format(path)
            )
        top = max(0, int(np.percentile(caption_tops, 5)) - margin_pixels)
        # Cropping uses an even output height; round the band up to make sure
        # the detected text is never left on the edge.
        top -= top % 2
        percent = 100 * (height - top) / height
        if percent >= 60:
            raise ValueError("Detected band is implausibly large for " + str(path))
        return BandEstimate(width, height, duration, sampled,
                            len(caption_tops), top, percent)
    finally:
        capture.release()
