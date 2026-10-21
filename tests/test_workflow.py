import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import clean_subtitles as cli
from subtitle_detection import BandEstimate, CaptionObservation, changing_caption_tracks


def estimate(present):
    return BandEstimate(640, 360, 10, 10, 6 if present else 0,
                        300 if present else None, 16.6667 if present else None,
                        has_subtitles=present)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / 'inputs'
        self.inputs.mkdir()
        self.output = self.root / 'outputs'

    def source(self, name='clip.mp4'):
        path = self.inputs / name
        path.write_bytes(b'original video bytes: ' + name.encode())
        return path

    @patch.object(cli, 'detect_bottom_band', return_value=estimate(False))
    @patch.object(cli.subprocess, 'run')
    def test_negative_renames_without_encoder_even_with_manual_band(self, run, detect):
        source = self.source()
        before = digest(source)
        result = cli.main([str(source), str(self.output), '--bottom-percent', '20',
                           '--ffmpeg', '/not-installed'])
        self.assertEqual(result, 0)
        self.assertEqual(digest(self.inputs / 'clip_no_subtitle.mp4'), before)
        self.assertFalse(source.exists())
        self.assertEqual(list(self.output.glob('*.mp4')), [])
        run.assert_not_called()
        report = json.loads((self.output / 'clip_no_subtitle.json').read_text())
        self.assertFalse(report['processed'])
        self.assertFalse(report['has_subtitles'])

    @patch.object(cli, 'detect_bottom_band', return_value=estimate(False))
    def test_tagging_is_idempotent_and_original_collision_is_preserved(self, detect):
        source = self.source('clip_no_subtitle.mp4')
        before = digest(source)
        self.assertEqual(cli.main([str(source), str(self.output)]), 0)
        self.assertEqual(digest(source), before)
        untagged = self.source()
        self.assertEqual(cli.main([str(untagged), str(self.output), '--overwrite']), 1)
        self.assertTrue(untagged.exists())
        self.assertEqual(digest(source), before)

    @patch.object(cli, 'detect_bottom_band', return_value=estimate(True))
    @patch.object(cli, 'ffmpeg_executable', return_value='ffmpeg')
    def test_positive_original_and_both_processed_names(self, ffmpeg, detect):
        for mode, suffix in [('crop', 'cropped'), ('blur', 'blurred')]:
            source = self.source('clip_' + mode + '_with_subtitle.mp4')
            before = digest(source)
            def encode(command, **kwargs):
                Path(command[-1]).write_bytes(b'encoded output')
            with patch.object(cli.subprocess, 'run', side_effect=encode):
                self.assertEqual(cli.main([str(source), str(self.output), '--mode', mode]), 0)
            self.assertEqual(digest(source), before)
            self.assertTrue((self.output / ('clip_' + mode + '_subtitle_' + suffix + '.mp4')).exists())
            self.assertFalse((self.output / ('clip_' + mode + '_with_subtitle_subtitle_' + suffix + '.mp4')).exists())

    @patch.object(cli, 'detect_bottom_band', return_value=estimate(True))
    @patch.object(cli.subprocess, 'run')
    def test_classify_only_keeps_original_bytes(self, run, detect):
        source = self.source()
        before = digest(source)
        self.assertEqual(cli.main([str(source), str(self.output), '--classify-only']), 0)
        self.assertEqual(digest(self.inputs / 'clip_with_subtitle.mp4'), before)
        run.assert_not_called()

    @patch.object(cli, 'detect_bottom_band', return_value=estimate(True))
    @patch.object(cli, 'ffmpeg_executable', return_value='ffmpeg')
    def test_failed_encoding_leaves_original_name_and_no_partial_output(self, ffmpeg, detect):
        source = self.source()
        with patch.object(cli.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['ffmpeg'])):
            self.assertEqual(cli.main([str(source), str(self.output)]), 1)
        self.assertTrue(source.exists())
        self.assertEqual(list(self.output.iterdir()), [])

    def test_decode_error_is_not_no_subtitle_and_batch_continues(self):
        bad = self.source('a_bad.mp4')
        self.source('z_good.mp4')
        def detect(path, *args, **kwargs):
            if path == bad:
                raise ValueError('Cannot decode video')
            return estimate(False)
        with patch.object(cli, 'detect_bottom_band', side_effect=detect):
            self.assertEqual(cli.main([str(self.inputs), str(self.output)]), 1)
        self.assertTrue(bad.exists())
        self.assertTrue((self.inputs / 'z_good_no_subtitle.mp4').exists())
        self.assertFalse((self.inputs / 'a_bad_no_subtitle.mp4').exists())

    def test_directory_does_not_reprocess_its_generated_outputs(self):
        source = self.source()
        self.source('clip_subtitle_cropped.mp4')
        self.source('clip_subtitle_blurred.mp4')
        self.assertEqual(cli.inputs_at(self.inputs), [source])


class CaptionEvidenceTests(unittest.TestCase):
    def observation(self, text, bottom=340):
        return CaptionObservation(1, bottom - 25, bottom, 220, 420, text)

    def test_static_nameplate_with_small_ocr_errors_is_not_dialogue(self):
        rows = [self.observation(text) for text in ['Tom Mackenzie', 'Tom Mackenzle', 'Tom Mackenzie']]
        self.assertEqual(changing_caption_tracks(rows, 640, 360)[0], [])

    def test_changing_captions_on_same_baseline_are_detected(self):
        rows = [self.observation(text) for text in ['Where are you going?', 'Back to work.', 'See you tomorrow.']]
        tracks, distinct = changing_caption_tracks(rows, 640, 360)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(distinct, 3)

    def test_unrelated_text_at_different_positions_is_not_one_track(self):
        rows = [self.observation('Clock', 305), self.observation('Name', 325), self.observation('Sign', 350)]
        self.assertEqual(changing_caption_tracks(rows, 640, 360)[0], [])


if __name__ == '__main__':
    unittest.main()
