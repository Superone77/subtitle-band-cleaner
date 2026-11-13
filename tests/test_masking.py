from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from subtitle_detection import BandEstimate, CaptionObservation, changing_caption_tracks
import subtitle_masking
from subtitle_masking import (caption_lines, caption_samples, fill_boxes, mask_command,
                              mask_timeline, mask_video, source_timing)


class FakeResult:
    def __init__(self, rows):
        self.boxes = [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]] for x0, y0, x1, y1, _, _ in rows] or None
        self.scores = [row[4] for row in rows]
        self.txts = [row[5] for row in rows]


TRACKS = [{"baseline": 340.0, "center_x": 320.0, "captions": 5}]


class CaptionLineTests(unittest.TestCase):
    def lines(self, rows, crop_top=0, tracks=TRACKS):
        return caption_lines(FakeResult(rows), crop_top, 640, 360, tracks, 1.0)

    def test_keeps_track_lines_and_drops_other_text(self):
        boxes, merged = self.lines([
            (220, 315, 420, 340, 0.95, 'Back to work.'),
            (240, 290, 400, 312, 0.95, 'Second line'),
            (250, 200, 390, 225, 0.95, 'Shop sign'),
            (20, 318, 120, 340, 0.95, 'Logo text'),
            (230, 316, 410, 339, 0.50, 'Blurry'),
            (300, 316, 330, 339, 0.99, 'A'),
        ])
        self.assertEqual(boxes, [(240, 290, 400, 312), (220, 315, 420, 340)])
        self.assertEqual(merged.text, 'Second line Back to work.')
        self.assertEqual((merged.top, merged.bottom), (290, 340))

    def test_second_line_below_a_top_anchored_caption_is_kept(self):
        tracks = [{"baseline": 320.0, "center_x": 320.0, "captions": 11}]
        boxes, merged = self.lines([
            (42, 272, 595, 319, 0.99, 'The word of the day is social.'),
            (245, 308, 397, 353, 0.98, 'Social -'),
            (431, 20, 593, 45, 0.95, 'Channel logo'),
        ], tracks=tracks)
        self.assertEqual(boxes, [(42, 272, 595, 319), (245, 308, 397, 353)])
        self.assertEqual((merged.top, merged.bottom), (272, 319))
        self.assertEqual(merged.text, 'The word of the day is social. Social -')

    def test_tall_line_on_the_track_is_kept(self):
        tracks = [{"baseline": 324.0, "center_x": 319.0, "captions": 11}]
        boxes, _ = self.lines([(250, 267, 382, 327, 1.0, 'Sorry.')], tracks=tracks)
        self.assertEqual(boxes, [(250, 267, 382, 327)])
        self.assertEqual(self.lines([(250, 250, 382, 327, 1.0, 'Sorry.')], tracks=tracks), ([], None))

    def test_crop_offset_is_added_back(self):
        boxes, merged = self.lines([(220, 15, 420, 40, 0.95, 'Back to work.')], crop_top=300)
        self.assertEqual(boxes, [(220, 315, 420, 340)])
        self.assertEqual(merged.bottom, 340)

    def test_box_reaching_into_ocr_padding_is_clipped_to_frame(self):
        tracks = [{"baseline": 357.0, "center_x": 320.0, "captions": 5}]
        boxes, merged = self.lines([(220, 30, 420, 66, 0.95, 'Back to work.')], crop_top=300, tracks=tracks)
        self.assertEqual(boxes, [(220, 330, 420, 360)])
        self.assertEqual(merged.bottom, 360)

    def test_text_at_another_baseline_is_not_a_caption(self):
        self.assertEqual(self.lines([(220, 260, 420, 285, 0.95, 'Scene text')]), ([], None))
        self.assertEqual(self.lines([]), ([], None))

    def test_forced_mode_keeps_every_candidate(self):
        boxes, _ = self.lines([
            (220, 315, 420, 340, 0.95, 'Back to work.'),
            (250, 200, 390, 225, 0.95, 'Shop sign'),
            (300, 316, 330, 339, 0.99, 'A'),
        ], tracks=[])
        self.assertEqual(boxes, [(250, 200, 390, 225), (220, 315, 420, 340)])


class StaticTextTests(unittest.TestCase):
    def sample(self, text, bottom):
        box = (220, bottom - 25, 420, bottom)
        return [box], CaptionObservation(1, bottom - 25, bottom, 220, 420, text)

    def test_unchanging_text_is_dropped_and_captions_kept(self):
        samples = [self.sample('Tom Mackenzie', 300) for _ in range(4)]
        samples += [self.sample(text, 340) for text in ['Where are you?', 'Back to work.', 'See you tomorrow.']]
        samples.append(([], None))
        kept = caption_samples(samples, 640, 360)
        self.assertEqual(kept[:4], [[], [], [], []])
        self.assertEqual(kept[4:7], [[(220, 315, 420, 340)]] * 3)
        self.assertEqual(kept[7], [])

    def test_same_text_at_another_position_starts_a_new_run(self):
        samples = [self.sample('Hello', 300)] * 3 + [self.sample('Hello', 340)] * 3
        samples += [self.sample(text, 340) for text in ['Back to work.', 'See you tomorrow.']]
        with patch('subtitle_masking.changing_caption_tracks', wraps=changing_caption_tracks) as tracks:
            kept = caption_samples(samples, 640, 360)
        self.assertEqual(len(tracks.call_args.args[0]), 4)
        self.assertEqual([bool(boxes) for boxes in kept], [False] * 3 + [True] * 5)

    def test_repeated_samples_of_one_caption_are_kept_together(self):
        samples = []
        for text in ['Where are you?', 'Back to work.', 'See you tomorrow.']:
            samples += [self.sample(text, 340)] * 5 + [([], None)]
        kept = caption_samples(samples, 640, 360)
        self.assertEqual([bool(boxes) for boxes in kept], ([True] * 5 + [False]) * 3)
        with patch('subtitle_masking.changing_caption_tracks', wraps=changing_caption_tracks) as tracks:
            caption_samples(samples, 640, 360)
        self.assertEqual(len(tracks.call_args.args[0]), 3)


class TimelineTests(unittest.TestCase):
    A, B, C = (1, 1, 2, 2), (3, 3, 4, 4), (5, 5, 6, 6)

    def test_look_ahead_and_last_sample(self):
        frames = mask_timeline([[], [self.A], [self.B]], 2, 8, hold_samples=0)
        self.assertEqual(frames[:2], [[self.A]] * 2)
        self.assertEqual(frames[2:4], [[self.A, self.B]] * 2)
        self.assertEqual(frames[4:], [[self.B]] * 4)

    def test_hold_across_missed_samples_then_stop(self):
        samples = [[self.A], [], [], [], [self.C]]
        frames = mask_timeline(samples, 1, 5, hold_samples=2)
        self.assertEqual(frames, [[self.A], [self.A], [self.A], [self.C], [self.C]])
        frames = mask_timeline(samples, 1, 5, hold_samples=1)
        self.assertEqual(frames, [[self.A], [self.A], [], [self.C], [self.C]])

    def test_union_is_deduplicated_and_no_samples_means_no_boxes(self):
        frames = mask_timeline([[self.A], [], [self.A]], 1, 3, hold_samples=1)
        self.assertEqual(frames, [[self.A], [self.A], [self.A]])
        self.assertEqual(mask_timeline([], 3, 2), [[], []])


class FillTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.random.default_rng(0).integers(0, 256, (120, 160, 3), dtype=np.uint8)

    def check(self, boxes, *regions):
        inside = np.zeros(self.frame.shape[:2], bool)
        for region in regions:
            inside[region] = True
        for fill in ('blur', 'pixelate', 'inpaint'):
            output = fill_boxes(self.frame, boxes, 4, fill)
            self.assertEqual(output.shape, self.frame.shape)
            self.assertTrue(np.array_equal(output[~inside], self.frame[~inside]), fill)
            changed = np.any(output != self.frame, axis=2)
            for region in regions:
                self.assertGreater(changed[region].mean(), 0.9, fill)

    def test_only_dilated_box_pixels_change(self):
        self.check([(40, 80, 120, 100)], np.s_[76:104, 36:124])

    def test_dilation_is_clipped_at_frame_edges(self):
        self.check([(0, 105, 60, 120), (130, 0, 160, 10)],
                   np.s_[101:120, 0:64], np.s_[0:14, 126:160])

    def test_frame_without_boxes_is_unchanged(self):
        for fill in ('blur', 'pixelate', 'inpaint'):
            self.assertEqual(fill_boxes(self.frame, [], 4, fill).tobytes(), self.frame.tobytes())

    def test_unknown_fill_is_rejected(self):
        with self.assertRaises(ValueError):
            fill_boxes(self.frame, [(1, 1, 5, 5)], 4, 'smudge')


class CommandAndEstimateTests(unittest.TestCase):
    def test_raw_frames_are_encoded_with_source_audio(self):
        command = mask_command('ffmpeg', 'in.mp4', 'out.mp4', 854, 480, 30000 / 1001, 18, 'veryfast')
        joined = ' '.join(command)
        self.assertIn('-f rawvideo -pixel_format bgr24 -video_size 854x480 -framerate 30000/1001 '
                      '-itsoffset 0.000000 -i - -i in.mp4', joined)
        self.assertIn('-map 0:v:0 -map 1:a? -sn -dn -c:v libx264', joined)
        self.assertIn('-crf 18 -preset veryfast -pix_fmt yuv420p -c:a copy', joined)
        self.assertEqual(command[-1], 'out.mp4')
        self.assertNotIn('setsar', joined)

    def test_source_start_and_pixel_shape_are_reapplied(self):
        command = mask_command('ffmpeg', 'in.mp4', 'out.mp4', 64, 64, 25, 18, 'medium', 1.0, '2:1')
        joined = ' '.join(command)
        self.assertIn('-itsoffset 1.000000 -i - -i in.mp4', joined)
        self.assertIn('-sn -dn -vf setsar=2/1 -c:v libx264', joined)

    def test_source_timing_parses_ffprobe_and_degrades_without_it(self):
        probe = subprocess.CompletedProcess([], 0, 'start_time=1.000000\nsample_aspect_ratio=2:1\n', '')
        with patch.object(subtitle_masking.shutil, 'which', return_value='/usr/bin/ffprobe'), \
                patch.object(subtitle_masking.Path, 'is_file', return_value=True), \
                patch.object(subtitle_masking.subprocess, 'run', return_value=probe) as run:
            self.assertEqual(source_timing('/usr/bin/ffmpeg', Path('in.mp4')), (1.0, '2:1'))
            self.assertEqual(run.call_args.args[0][-1], 'in.mp4')
            probe.stdout = 'start_time=N/A\nsample_aspect_ratio=1:1\n'
            self.assertEqual(source_timing('/usr/bin/ffmpeg', Path('in.mp4')), (0.0, None))
        with patch.object(subtitle_masking.shutil, 'which', return_value=None), \
                patch.object(subtitle_masking.Path, 'is_file', return_value=False):
            self.assertEqual(source_timing('/opt/ffmpeg', Path('in.mp4')), (0.0, None))

    def test_band_estimate_caption_tracks_default(self):
        first = BandEstimate(640, 360, 10, 10, 6, 300, 16.6667, True)
        second = BandEstimate(640, 360, 10, 10, 0, None, None, False)
        self.assertEqual(first.caption_tracks, [])
        self.assertIsNot(first.caption_tracks, second.caption_tracks)
        self.assertEqual(first.as_dict()['caption_tracks'], [])


if __name__ == '__main__':
    unittest.main()


class FakeCapture:
    """Sequential frames like cv2.VideoCapture; one instance per decode pass."""
    passes = []

    def __init__(self, path):
        self.frames = FakeCapture.passes.pop(0)
        self.index = 0

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2
        return {cv2.CAP_PROP_FPS: 10.0, cv2.CAP_PROP_FRAME_COUNT: 6.0,
                cv2.CAP_PROP_FRAME_WIDTH: 32.0, cv2.CAP_PROP_FRAME_HEIGHT: 24.0}[prop]

    def grab(self):
        self.index += 1
        return self.index <= len(self.frames)

    def retrieve(self):
        return True, self.frames[self.index - 1]

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        self.index += 1
        return True, self.frames[self.index - 1]

    def release(self):
        pass


class FakeProcess:
    def __init__(self, command, stdin=None, code=0):
        self.command, self.code, self.written, self.killed = command, code, bytearray(), False
        process = self

        class Stdin:
            def write(self, data):
                process.written += data

            def close(self):
                pass
        self.stdin = Stdin()

    def wait(self):
        return self.code

    def kill(self):
        self.killed = True


class FakeOCR:
    def __init__(self, rows):
        self.rows = rows

    def __call__(self, image):
        return FakeResult(self.rows)


class MaskVideoTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.frames = [rng.integers(0, 256, (24, 32, 3), dtype=np.uint8) for _ in range(6)]
        self.processes = []

    def run_mask(self, passes, rows=(), code=0):
        FakeCapture.passes = [list(frames) for frames in passes]
        def popen(command, stdin=None):
            self.processes.append(FakeProcess(command, stdin, code))
            return self.processes[-1]
        with patch('cv2.VideoCapture', FakeCapture), \
                patch.object(subtitle_masking.subprocess, 'Popen', side_effect=popen), \
                patch.object(subtitle_masking, 'source_timing', return_value=(0.0, None)):
            return mask_video(Path('in.mp4'), Path('out.mp4'), 'ffmpeg', 18, 'medium', 12, [],
                              sample_fps=5, hold_samples=0, margin_pixels=1, ocr=FakeOCR(list(rows)))

    def test_frames_without_captions_are_piped_through_unchanged(self):
        report = self.run_mask([self.frames, self.frames])
        self.assertEqual(bytes(self.processes[0].written), b''.join(frame.tobytes() for frame in self.frames))
        self.assertEqual((report['frames'], report['masked_frames'], report['samples']), (6, 0, 3))
        self.assertEqual(self.processes[0].command[-1], 'out.mp4')

    def test_caption_boxes_are_filled_only_where_sampled(self):
        report = self.run_mask([self.frames, self.frames], rows=[(8, 2, 28, 6, 0.95, 'Hello there')])
        written = np.frombuffer(bytes(self.processes[0].written), np.uint8).reshape(6, 24, 32, 3)
        self.assertEqual(report['masked_frames'], 6)
        self.assertTrue(np.array_equal(written[0, 19:], self.frames[0][19:]))
        self.assertTrue(np.array_equal(written[0, :12], self.frames[0][:12]))
        self.assertFalse(np.array_equal(written[0, 12:18, 7:29], self.frames[0][12:18, 7:29]))

    def test_short_second_pass_is_an_error_not_a_truncated_video(self):
        with self.assertRaises(ValueError):
            self.run_mask([self.frames, self.frames[:2]])
        with self.assertRaises(ValueError):
            self.run_mask([self.frames, self.frames + self.frames[:1]])
        self.assertTrue(self.processes[-1].killed)

    def test_encoder_failure_is_reported(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_mask([self.frames, self.frames], code=1)
