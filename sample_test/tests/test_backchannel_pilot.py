import copy
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_backchannel_pilot import overlay_pcm, extend_alignment, anchor_end
from build_backchannel_pilot import sha
from build_backchannel_voice_comparison import build_comparison


class BackchannelPilotTests(unittest.TestCase):
    def setUp(self):
        self.pcm = np.zeros((10000, 2), dtype=np.int16)
        self.pcm[:, 1] = np.arange(10000, dtype=np.int16)
        self.pcm[8000:, 0] = 500
        self.clip = np.full(500, 1000, dtype=np.int16)
        self.user = dict(start=0., end=7.)
        self.segments = [dict(speaker='ASSISTANT', start=8., end=10.)]

    def test_only_selected_channel_and_samples_change(self):
        original = self.pcm.copy()
        result = overlay_pcm(self.pcm, self.clip, 3000, 1000, self.segments, self.user)
        np.testing.assert_array_equal(self.pcm, original)
        np.testing.assert_array_equal(result[:, 1], original[:, 1])
        np.testing.assert_array_equal(result[:3000, 0], original[:3000, 0])
        np.testing.assert_array_equal(result[3500:, 0], original[3500:, 0])
        np.testing.assert_array_equal(result[3000:3500, 0], self.clip)

    def test_reject_nonzero_audio_in_guard(self):
        self.pcm[2800, 0] = 1
        with self.assertRaisesRegex(ValueError, 'not exactly silent'):
            overlay_pcm(self.pcm, self.clip, 3000, 1000, self.segments, self.user)

    def test_reject_user_end_and_assistant_collision(self):
        with self.assertRaises(ValueError):
            overlay_pcm(self.pcm, self.clip, 6500, 1000, self.segments, self.user)
        self.segments.append(dict(speaker='ASSISTANT', start=3.6, end=3.8))
        with self.assertRaisesRegex(ValueError, 'intersects'):
            overlay_pcm(self.pcm, self.clip, 3000, 1000, self.segments, self.user)

    def test_alignment_merge_does_not_recompute_originals(self):
        original = dict(alignments=[['元', [6.,7.], 'ASSISTANT']],
                        token_alignments=[[[100], [6.,7.], 'ASSISTANT']],
                        utterances=[dict(speaker='ASSISTANT',text='元',start=6.,end=7.)])
        before = copy.deepcopy(original)
        result, added = extend_alignment(original, [('うん',[3.,3.3],10)],'うん',3.,3.3)
        self.assertEqual(before,original)
        for key in original:
            self.assertEqual(result[key], added[key]+original[key])

    def test_anchor_requires_boundary_and_no_fabricated_timing(self):
        words=[dict(text='そう',absolute_end=1.),dict(text='ですね。',absolute_end=2.)]
        self.assertEqual(anchor_end(words,'そうですね。'),2.)
        with self.assertRaises(ValueError):anchor_end(words,'そうです')
        with self.assertRaises(ValueError):anchor_end(words,'ちがいます。')


class BackchannelComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / 'original'
        self.assets = self.root / 'assets'
        self.out = self.root / 'comparison'
        self.original.mkdir()
        self.assets.mkdir()
        pcm = np.zeros((10000, 2), dtype=np.int16)
        pcm[:, 1] = np.arange(10000, dtype=np.int16)
        self.paths = {key: self.root / name for key, name in {
            'audio_path': 'original.wav', 'image_path': 'image.jpg',
            'speech_dialogue_path': 'dialogue.json', 'alignment_path': 'alignment.json'}.items()}
        sf.write(self.paths['audio_path'], pcm, 1000, subtype='PCM_16')
        self.paths['image_path'].write_bytes(b'synthetic image fixture; not used by audio builder')
        self.paths['alignment_path'].write_text('{}')
        user = dict(speaker='USER', text='架空のテスト対話です。', start=0., end=7.)
        self.paths['speech_dialogue_path'].write_text(json.dumps({'segments': [user]}))
        hashes = {key: sha(path) for key, path in self.paths.items()}
        source = {key: str(path) for key, path in self.paths.items()}
        source.update(uid='synthetic-test', segment_index=0, segment=user)
        plan = {'samples': [{'source': source, 'source_sha256': hashes}]}
        (self.original / 'plan.json').write_text(json.dumps(plan))
        item = dict(uid='synthetic-test', source_sha256=hashes,
                    event=dict(text='うん。', audible_start=3., start=3., end=3.5),
                    snippet_start=0., user_text=user['text'], anchor_prefix='架空', title='テスト')
        for key in ['image_url', 'original_url', 'augmented_url', 'snippet_original_url', 'snippet_augmented_url']:
            item[key] = 'fixture.flac'
        (self.original / 'manifest.json').write_text(json.dumps({'samples': [item]}))
        timings = {}
        for name in ['un_expressive', 'unun_neutral', 'unun_expressive']:
            path = self.assets / f'{name}.flac'
            sf.write(path, np.full(500, 1000, dtype=np.int16), 1000, subtype='PCM_16')
            timings[name] = dict(audio_sha256=sha(path), transcript='うん')
        (self.assets / 'timings.json').write_text(json.dumps(timings))

    def test_rejects_baseline_source_changed_since_pilot(self):
        self.paths['image_path'].write_bytes(b'changed image')
        with self.assertRaisesRegex(ValueError, 'original pilot'):
            build_comparison(self.out, self.assets, self.original)
        self.assertFalse(self.out.exists())

    def test_comparison_preserves_original_pcm_and_refuses_overwrite(self):
        hashes = {key: sha(path) for key, path in self.paths.items()}
        with contextlib.redirect_stdout(io.StringIO()):
            page = build_comparison(self.out, self.assets, self.original)
        self.assertTrue(page.is_file())
        self.assertEqual(hashes, {key: sha(path) for key, path in self.paths.items()})
        result = json.loads((self.out / 'manifest.json').read_text())
        self.assertEqual(len(result['samples']), 4)
        for sample in result['samples'][1:]:
            self.assertTrue(all(sample['validation'].values()))
            self.assertFalse(sample['training_eligible'])
        with self.assertRaises(FileExistsError):
            build_comparison(self.out, self.assets, self.original)


if __name__ == '__main__':
    unittest.main()
