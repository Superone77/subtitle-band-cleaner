"""Mask detected caption text boxes only in frames where captions are shown."""

from fractions import Fraction
from math import ceil, floor

from subtitle_detection import CaptionObservation, changing_caption_tracks, normalized_text
from video_export import DEFAULT_MAX_HEIGHT, export_size


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



def displayed_array(frame):
    import numpy as np

    array = frame.to_ndarray(format="bgr24")
    return np.ascontiguousarray(np.rot90(array, frame.rotation // 90)) if frame.rotation else array


def yuv_with_boxes(frame, boxes, margin_pixels, fill, colorspace, color_range):
    """Copy the converted mask only into its affected luma/chroma rectangles."""
    import av
    import numpy as np

    base = frame.reformat(format="yuv420p")
    if not boxes and not frame.rotation:
        return base
    pixels = displayed_array(frame)
    if boxes:
        pixels = fill_boxes(pixels, boxes, margin_pixels, fill)
    converted = av.VideoFrame.from_ndarray(pixels, format="bgr24").reformat(
        format="yuv420p", dst_colorspace=colorspace, dst_color_range=color_range)
    if frame.rotation:
        return converted
    # Decoder-owned planes can be referenced by later frames, so do not
    # mutate base. The new frame owns all of its planes.
    output = av.VideoFrame(base.width, base.height, "yuv420p")
    for index, (old, patch, target) in enumerate(zip(base.planes, converted.planes, output.planes)):
        def view(plane):
            return np.ndarray((plane.height, plane.line_size), np.uint8,
                              buffer=plane)[:, :plane.width]
        region, replacement = view(target), view(patch)
        region[:] = view(old)
        factor = 2 if index else 1
        for x0, y0, x1, y1 in boxes:
            area = np.s_[max(0, (y0 - margin_pixels) // factor):ceil((y1 + margin_pixels) / factor),
                         max(0, (x0 - margin_pixels) // factor):ceil((x1 + margin_pixels) / factor)]
            region[area] = replacement[area]
    return output


def write_masked_video(source, destination, timeline, crf, preset, margin_pixels, fill,
                       max_height, native, turned):
    """Encode with the original video PTS and mux untouched source audio packets."""
    import av
    from av.video.reformatter import ColorRange, Colorspace

    options = {"movflags": "+faststart", "avoid_negative_ts": "disabled",
               "movie_timescale": "1000000"}
    with av.open(str(source)) as src, av.open(str(destination), "w", options=options) as out:
        video = src.streams.video[0]
        codec = video.codec_context
        rate = video.average_rate or video.guessed_rate or Fraction(25)
        encoded = out.add_stream("libx264", rate=rate)
        encoded.width, encoded.height = export_size(*native, max_height)
        encoded.codec_context.time_base = encoded.time_base = video.time_base
        encoded.options = {"crf": str(crf), "preset": preset}
        encoded.pix_fmt = "yuv420p"
        for name in ("color_range", "colorspace", "color_trc", "color_primaries"):
            setattr(encoded.codec_context, name, getattr(codec, name))
        try:
            colorspace = Colorspace(codec.colorspace)
        except ValueError:
            colorspace = None
        color_range = ColorRange(codec.color_range) if codec.color_range in (1, 2) else None
        sar = Fraction(video.sample_aspect_ratio or 1)
        width, height = native
        encoded.codec_context.sample_aspect_ratio = (
            (1 / sar if turned else sar) * Fraction(width * encoded.height, height * encoded.width))
        audio = {stream.index: out.add_stream_from_template(stream) for stream in src.streams.audio}
        durations = {}
        frames = 0

        def mux(packets):
            for packet in packets:
                duration = durations.pop(packet.pts, 0)
                if duration > 0:
                    packet.duration = duration
                out.mux(packet)

        for packet in src.demux(video, *src.streams.audio):
            if packet.stream.index in audio:
                if packet.dts is not None:
                    packet.stream = audio[packet.stream.index]
                    out.mux(packet)
                continue
            for frame in packet.decode():
                if frames >= len(timeline):
                    raise ValueError("Second decode pass returned more frames than the first: " + str(source))
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("Video frame has no presentation timestamp: " + str(source))
                rotated = frame.rotation % 180 != 0
                size = (frame.height, frame.width) if rotated else (frame.width, frame.height)
                if size != native:
                    raise ValueError("Frame size changed while decoding: " + str(source))
                pts = round(frame.pts * frame.time_base / video.time_base)
                duration = round(frame.duration * frame.time_base / video.time_base)
                output = yuv_with_boxes(frame, timeline[frames], margin_pixels, fill, colorspace, color_range)
                output = output.reformat(encoded.width, encoded.height, "yuv420p",
                                         interpolation="LANCZOS")
                output.pts, output.time_base, output.duration = pts, video.time_base, duration
                durations[pts] = duration
                mux(encoded.encode(output))
                frames += 1
        if frames != len(timeline):
            raise ValueError("Second decode pass returned {} of {} frames: {}".format(
                frames, len(timeline), source))
        mux(encoded.encode())
        return frames, encoded.width, encoded.height


def mask_video(source, destination, crf, preset, band_top, caption_tracks,
               sample_fps=5.0, hold_samples=3, margin_pixels=8, fill="blur",
               min_confidence=0.80, min_distinct_captions=3, alignment="center", ocr=None,
               max_height=DEFAULT_MAX_HEIGHT):
    """OCR the band, then re-encode with caption boxes masked frame by frame."""
    import av
    import cv2

    if sample_fps <= 0 or hold_samples < 0 or margin_pixels < 0 or min_distinct_captions < 1:
        raise ValueError("Invalid masking parameters")
    if fill not in FILLS:
        raise ValueError("Fill must be one of: " + ", ".join(FILLS))
    if alignment not in {"center", "any"}:
        raise ValueError("Alignment must be center or any")
    if ocr is None:
        from rapidocr import RapidOCR

        ocr = RapidOCR(params={"Det.limit_type": "max", "Det.limit_side_len": 960,
                               "Global.log_level": "error"})
    samples = []
    frames = 0
    with av.open(str(source)) as capture:
        if not capture.streams.video:
            raise ValueError("Cannot decode video: " + str(source))
        stream = capture.streams.video[0]
        step = max(1, round(float(stream.average_rate or stream.guessed_rate or 25) / sample_fps))
        crop_top = max(0, band_top - margin_pixels)
        print("{}: mask pass 1/2, OCR every {} frames below y={}".format(
            source.name, step, crop_top), flush=True)
        for frame in capture.decode(stream):
            if frames % step == 0:
                pixels = displayed_array(frame)
                height, width = pixels.shape[:2]
                if frames == 0:
                    native = width, height
                    turned = frame.rotation % 180 != 0
                if not 0 <= band_top < height:
                    raise ValueError("Band top is outside the frame: " + str(band_top))
                band = cv2.copyMakeBorder(pixels[crop_top:, :], 0, OCR_PADDING, 0, 0,
                                          cv2.BORDER_CONSTANT, value=0)
                samples.append(caption_lines(
                    ocr(band), crop_top, width, height, caption_tracks,
                    frame.time or 0.0, min_confidence, alignment))
            frames += 1
        if frames < max(1, ceil(stream.frames * 0.8)):
            raise ValueError("Too many frames could not be decoded: " + str(source))
    sample_boxes = (caption_samples(samples, width, height, min_distinct_captions)
                    if caption_tracks else [boxes for boxes, _ in samples])
    timeline = mask_timeline(sample_boxes, step, frames, hold_samples)
    masked_frames = sum(bool(boxes) for boxes in timeline)
    print("{}: mask pass 2/2, {} on {} of {} frames".format(
        source.name, fill, masked_frames, frames), flush=True)
    frames, output_width, output_height = write_masked_video(
        source, destination, timeline, crf, preset, margin_pixels, fill, max_height, native, turned)
    return {
        "fill": fill,
        "sample_fps": sample_fps,
        "hold_samples": hold_samples,
        "samples": len(samples),
        "samples_with_captions": sum(bool(boxes) for boxes in sample_boxes),
        "frames": frames,
        "masked_frames": masked_frames,
        "width": output_width,
        "height": output_height,
    }
