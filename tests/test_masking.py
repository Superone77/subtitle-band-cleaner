import unittest
from unittest.mock import patch

import numpy as np

from subtitle_detection import BandEstimate, CaptionObservation, changing_caption_tracks
from subtitle_masking import caption_lines, caption_samples, fill_boxes, mask_timeline


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


class BandEstimateTests(unittest.TestCase):
    def test_band_estimate_caption_tracks_default(self):
        first = BandEstimate(640, 360, 10, 10, 6, 300, 16.6667, True)
        second = BandEstimate(640, 360, 10, 10, 0, None, None, False)
        self.assertEqual(first.caption_tracks, [])
        self.assertIsNot(first.caption_tracks, second.caption_tracks)
        self.assertEqual(first.as_dict()['caption_tracks'], [])


if __name__ == '__main__':
    unittest.main()
