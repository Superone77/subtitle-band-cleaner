"""Exercise actual MP4 encoding, VFR PTS, delayed video and audio copying."""

from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import subprocess
import tempfile
import unittest

import av
import numpy as np

from clean_subtitles import ffmpeg_executable, make_command
from subtitle_masking import mask_video, write_masked_video, yuv_with_boxes


class EmptyOCR:
    boxes = None

    def __call__(self, image):
        return self


def media_timing(path):
    with av.open(str(path)) as source:
        audio = [(p.pts * p.time_base, p.duration * p.time_base, sha256(bytes(p)).digest())
                 for p in source.demux(audio=0) if p.pts is not None]
    with av.open(str(path)) as source:
        stream = source.streams.video[0]
        frames = [f.pts * f.time_base for f in source.decode(stream)]
        return (stream.width, stream.height), frames, audio


class MP4ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        cls.ffmpeg = ffmpeg_executable(None)
        cls.sources = {}
        for h in (360, 480, 720):
            p = cls.root / f'in-{h}.mp4'
            # Two different cadences and video starting 400 ms after the audio.
            subprocess.run([
                cls.ffmpeg, '-v', 'error', '-copyts',
                '-f', 'lavfi', '-i', f'testsrc2=s={h * 4 // 3}x{h}:r=25:d=1.6',
                '-f', 'lavfi', '-i', 'sine=frequency=773:sample_rate=48000:duration=2',
                '-vf', "select='lt(n,15)+not(mod(n,2))',setpts=PTS+10",
                '-fps_mode', 'passthrough', '-enc_time_base:v', 'filter',
                '-c:v', 'libx264', '-c:a', 'aac', str(p),
            ], check=True)
            cls.sources[h] = p

    def test_export_caps_height_without_upscaling_and_preserves_timing(self):
        for h, source in self.sources.items():
            _, before, audio = media_timing(source)
            self.assertEqual(before[0], Fraction(2, 5))
            self.assertGreater(len(set(b - a for a, b in zip(before, before[1:]))), 1)
            for mode in ('crop', 'blur', 'mask'):
                with self.subTest(height=h, mode=mode):
                    output = self.root / f'{h}-{mode}.mp4'
                    if mode == 'mask':
                        mask_video(source, output, 16, 'veryfast', h * 3 // 4, [],
                                   ocr=EmptyOCR())
                    else:
                        subprocess.run(make_command(self.ffmpeg, source, output,
                                                    mode, 20, 16, 'veryfast'), check=True,
                                       stderr=subprocess.PIPE)
                    size, actual, copied_audio = media_timing(output)
                    kept = h * 4 // 5 if mode == 'crop' else h
                    self.assertEqual(size[1], min(480, kept))
                    self.assertEqual(actual, before)
                    self.assertEqual(copied_audio, audio)

    def test_mask_rejects_incomplete_frame_timeline(self):
        source = self.sources[360]
        with self.assertRaisesRegex(ValueError, 'more frames'):
            write_masked_video(source, self.root / 'short.mp4', [[]], 16, 'veryfast',
                               8, 'blur', 480, (480, 360), False)

    def test_mask_keeps_original_yuv_samples_outside_the_box_before_encoding(self):
        with av.open(str(self.sources[360])) as source:
            frame = next(source.decode(video=0))
            original = frame.to_ndarray()
            changed = yuv_with_boxes(frame, [(110, 70, 230, 100)], 4, 'blur', None, None)
            result = changed.to_ndarray()
            self.assertFalse(np.array_equal(original, result))
            self.assertTrue(np.array_equal(original[:66], result[:66]))
            for before, after in zip(frame.planes, changed.planes):
                self.assertEqual(bytes(before)[:before.line_size], bytes(after)[:after.line_size])
