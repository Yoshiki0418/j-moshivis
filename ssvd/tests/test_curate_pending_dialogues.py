import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from curate_pending_dialogues import validate
import curate_pending_dialogues as curator
from ai_suggestion_store import default_ai_suggestion_store
from review_store import ReviewStore, content_fingerprint


class CodexExecutableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.binary = Path(self.temp.name) / 'codex'
        self.binary.write_text('#!/bin/sh\nexit 0\n')
        self.binary.chmod(0o700)

    def test_discovers_executable_on_path(self):
        with mock.patch.dict(os.environ, {'PATH': self.temp.name}):
            self.assertEqual(curator.resolve_codex(), str(self.binary.resolve()))

    def test_explicit_path_works_without_path_entry(self):
        with mock.patch.dict(os.environ, {'PATH': ''}):
            self.assertEqual(curator.resolve_codex(str(self.binary)), str(self.binary.resolve()))

    def test_missing_explicit_binary_does_not_silently_use_path(self):
        with mock.patch.dict(os.environ, {'PATH': self.temp.name}):
            with self.assertRaisesRegex(FileNotFoundError, '--codex /path/to/codex'):
                curator.resolve_codex(str(Path(self.temp.name) / 'missing-codex'))
        with mock.patch.dict(os.environ, {'PATH': ''}):
            with self.assertRaisesRegex(FileNotFoundError, 'Add codex to PATH'):
                curator.resolve_codex()


class CorrectionValidationTests(unittest.TestCase):
    def setUp(self):
        self.record={'dialogue':[{'speaker':'ユーザー','text':'何人いますか？'},
                                 {'speaker':'アシスタント','text':'3人です。'}],
                     'auxiliary_task':{'kind':'counting','answer_count':3,'counted_target':'人',
                                       'question_turn_index':0,'answer_turn_index':1,'grounding_source':'vlm_visual'}}
        self.response={'turns':[{'turn_index':0,'text':'何人いますか？'},{'turn_index':1,'text':'2人です。'}],
                       'auxiliary_update':{'answer_count':2,'counted_target':'人','recognized_text':None},
                       'changes':[{'turn_index':1,'reason':'画像では2人'}], 'requires_attention':False,'concerns':[]}

    def test_edits_preserve_roles_and_annotation_provenance(self):
        old=copy.deepcopy(self.record)
        d=validate(self.response,self.record)
        self.assertEqual(d['dialogue'][0]['speaker'],'ユーザー')
        self.assertEqual(d['auxiliary_task']['answer_count'],2)
        self.assertEqual(d['auxiliary_task']['grounding_source'],'vlm_visual')
        self.assertEqual(d['verdict'],'corrected')
        self.assertEqual(old,self.record)

    def test_cannot_add_drop_or_reorder_turns(self):
        for turns in [[],list(reversed(self.response['turns'])),self.response['turns']*2]:
            d=copy.deepcopy(self.response);d['turns']=turns
            with self.assertRaises(ValueError):validate(d,self.record)

    def test_invalid_count_and_wrong_task_rejected(self):
        for count in [0,-1,1001,True,'2']:
            d=copy.deepcopy(self.response);d['auxiliary_update']['answer_count']=count
            with self.assertRaises(ValueError):validate(d,self.record)
        d=copy.deepcopy(self.response);d['auxiliary_update']['recognized_text']='店'
        with self.assertRaises(ValueError):validate(d,self.record)

    def test_uncertain_annotation_is_never_claimed_resolved(self):
        self.response['auxiliary_update']['answer_count']=None
        d=validate(self.response,self.record)
        self.assertEqual(d['verdict'],'needs_attention')
        self.assertTrue(d['concerns'])

    def test_identical_output_is_unchanged(self):
        self.response['turns'][1]['text']='3人です。'
        self.response['auxiliary_update']['answer_count']=3
        d=validate(self.response,self.record)
        self.assertEqual(d['verdict'],'unchanged')
        self.assertEqual(d['changes'],[])


class CurationIsolationTests(unittest.TestCase):
    """Exercise a temporary real catalog/queue without launching any model."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = self.root / 'catalog.sqlite3'
        self.run = self.root / 'run'
        self.review = ReviewStore(self.root / 'data/curation/dialogue_review_events.jsonl')
        self.source_count = 0
        for name, value in (('ROOT', self.root), ('CATALOG', self.catalog), ('STOP', threading.Event())):
            patcher = mock.patch.object(curator, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        with curator.connect(self.catalog) as con:
            con.execute('CREATE TABLE sources(source_id INTEGER PRIMARY KEY,path TEXT)')
            con.execute('''CREATE TABLE items(
                collection TEXT,uid TEXT,category TEXT,review_required INTEGER,
                review_status TEXT,has_dialogue INTEGER,has_image INTEGER,
                summary_json TEXT,dialogue_source INTEGER,dialogue_offset INTEGER,
                dialogue_length INTEGER,review_id TEXT,image_path TEXT,
                PRIMARY KEY(collection,uid))''')

    def add_item(self, uid='target', collection='visual', **overrides):
        self.source_count += 1
        index = self.source_count
        image = self.root / f'image_{index}.jpg'
        image.write_bytes(b'fake-image-bytes-for-hash-only')
        image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
        record = {'uid': uid, 'image': image.name,
                  'dialogue': [{'speaker': 'ユーザー', 'text': '何が見えますか？'},
                               {'speaker': 'アシスタント', 'text': '犬が見えます。'}],
                  'auxiliary_task': None, 'verification_status': 'unverified_vlm_visual'}
        source = self.root / f'source_{index}.jsonl'
        raw = json.dumps(record, ensure_ascii=False).encode() + b'\n'
        source.write_bytes(raw)
        fingerprint = content_fingerprint(record, image_sha256=image_hash)
        row = dict(collection=collection, uid=uid, category='normal', review_required=1,
                   review_status='pending', has_dialogue=1, has_image=1,
                   summary_json=json.dumps({'image_integrity_status': 'verified',
                                            'current_image_sha256': image_hash}),
                   dialogue_source=index, dialogue_offset=0, dialogue_length=len(raw),
                   review_id=fingerprint, image_path=image.name)
        row.update(overrides)
        with curator.connect(self.catalog) as con:
            con.execute('INSERT INTO sources VALUES (?,?)', (index, source.name))
            con.execute('INSERT INTO items (' + ','.join(row) + ') VALUES (' + ','.join('?' for _ in row) + ')', tuple(row.values()))
        return dict(row=row, record=record, source=source, image=image, image_hash=image_hash)

    def snapshot(self):
        with redirect_stdout(io.StringIO()):
            curator.snapshot(self.run)
        with curator.connect(self.run / 'queue.sqlite3', True) as con:
            return [dict(row) for row in con.execute('SELECT * FROM jobs ORDER BY collection,uid')]

    def decision(self, item, decision='pending'):
        row = item['row']
        self.review.append(fingerprint=row['review_id'], decision=decision,
                           collection=row['collection'], uid=row['uid'])

    @staticmethod
    def response(record):
        turns = [{'turn_index': i, 'text': turn['text']} for i, turn in enumerate(record['dialogue'])]
        turns[1]['text'] = '猫が見えます。'
        return {'turns': turns, 'auxiliary_update': {'answer_count': None, 'counted_target': None, 'recognized_text': None},
                'changes': [{'turn_index': 1, 'reason': '画像と一致する対象に修正'}],
                'requires_attention': False, 'concerns': []}

    def run_mock_worker(self, hook=None, workers=1):
        def invoke(executable, job, artifact, timeout):
            if hook:
                hook(job, artifact)
            return self.response(json.loads(job['record_json'])), {'model': 'gpt-6-luna', 'elapsed_seconds': 0.01}
        with mock.patch.object(curator, 'invoke_codex', side_effect=invoke) as invocation:
            with redirect_stdout(io.StringIO()):
                curator.run_jobs(self.run, 'never-executed-codex', workers, 0, 1)
        return invocation

    def test_snapshot_includes_only_verified_required_untouched_pending_rows(self):
        self.add_item('eligible')
        for name, values in (
                ('not_required', {'review_required': 0}), ('approved_catalog', {'review_status': 'approved'}),
                ('no_dialogue', {'has_dialogue': 0}), ('no_image', {'has_image': 0}),
                ('edited', {'summary_json': json.dumps({'human_revision': True, 'image_integrity_status': 'verified'})}),
                ('bad_image', {'summary_json': json.dumps({'image_integrity_status': 'mismatch'})}),
                ('stale_catalog', {'review_id': 'f' * 64})):
            self.add_item(name, **values)
        for decision in ('pending', 'approved', 'rejected'):
            item = self.add_item('human_' + decision)
            self.decision(item, decision)
        before = self.review.path.read_bytes()
        jobs = self.snapshot()
        self.assertEqual(['eligible'], [job['uid'] for job in jobs])
        self.assertEqual(before, self.review.path.read_bytes())

    def test_snapshot_excludes_all_collections_touched_at_same_fingerprint(self):
        first = self.add_item('same', collection='first')
        # Two collections legitimately share the same source record and image.
        row = dict(first['row'], collection='second')
        with curator.connect(self.catalog) as con:
            con.execute('INSERT INTO items (' + ','.join(row) + ') VALUES (' + ','.join('?' for _ in row) + ')', tuple(row.values()))
        self.decision(first, 'approved')
        self.review.append(fingerprint=row['review_id'], decision='pending', collection='second', uid='same')
        self.assertEqual([], self.snapshot())

    def test_eligible_rechecks_each_human_decision_after_snapshot(self):
        items = [self.add_item(decision) for decision in ('pending', 'approved', 'rejected')]
        jobs = {job['uid']: job for job in self.snapshot()}
        for item in items:
            decision = item['row']['uid']
            self.decision(item, decision)
            self.assertEqual((False, 'human_review_exists'), curator.eligible(jobs[decision]))

    def test_eligible_detects_changed_source_even_with_stale_catalog(self):
        item = self.add_item()
        job = self.snapshot()[0]
        self.assertEqual((True, None), curator.eligible(job))
        item['source'].write_bytes(item['source'].read_bytes().replace('犬'.encode(), '猫'.encode()))
        self.assertEqual((False, 'source_content_changed'), curator.eligible(job))

    def test_eligible_detects_changed_image_without_catalog_update(self):
        item = self.add_item()
        job = self.snapshot()[0]
        item['image'].write_bytes(b'different-image')
        self.assertEqual((False, 'image_changed'), curator.eligible(job))

    def test_eligible_detects_changed_catalog_fingerprint_and_image_path(self):
        item = self.add_item()
        job = self.snapshot()[0]
        with curator.connect(self.catalog) as con:
            con.execute('UPDATE items SET review_id=?', ('c' * 64,))
        self.assertEqual((False, 'source_fingerprint_changed'), curator.eligible(job))
        replacement = self.root / 'new_path.jpg'
        replacement.write_bytes(item['image'].read_bytes())
        with curator.connect(self.catalog) as con:
            con.execute('UPDATE items SET review_id=?,image_path=?', (job['base_fingerprint'], replacement.name))
        self.assertEqual((False, 'image_path_changed'), curator.eligible(job))

    def test_worker_saves_draft_without_changing_source_catalog_or_human_decisions(self):
        item = self.add_item()
        self.review.append(fingerprint='d' * 64, decision='approved', collection='unrelated', uid='unrelated')
        job = self.snapshot()[0]
        before = {path: path.read_bytes() for path in (item['source'], item['image'], self.catalog, self.review.path)}
        invocation = self.run_mock_worker()
        self.assertEqual(1, invocation.call_count)
        for path, raw in before.items():
            self.assertEqual(raw, path.read_bytes(), str(path))
        store = default_ai_suggestion_store(self.root)
        draft = store.get_suggestion(**{key: job[key] for key in ('collection', 'uid', 'base_fingerprint', 'image_sha256')})
        self.assertEqual('gpt-6-luna', draft['model'])
        self.assertEqual('corrected', draft['verdict'])
        self.assertNotIn('approved', draft)
        self.assertEqual((True, None), curator.eligible(job))
        self.assertEqual(1, store.count())

    def test_worker_discards_result_when_human_review_arrives_during_generation(self):
        item = self.add_item()
        self.snapshot()
        self.run_mock_worker(hook=lambda job, artifact: self.decision(item))
        self.assertEqual(0, default_ai_suggestion_store(self.root).count())
        with curator.connect(self.run / 'queue.sqlite3', True) as con:
            row = con.execute('SELECT state,error FROM jobs').fetchone()
        self.assertEqual(('skipped', 'human_review_exists'), tuple(row))

    def test_worker_discards_result_when_image_changes_during_generation(self):
        item = self.add_item()
        self.snapshot()
        self.run_mock_worker(hook=lambda job, artifact: item['image'].write_bytes(b'changed'))
        self.assertEqual(0, default_ai_suggestion_store(self.root).count())
        self.assertFalse(self.review.path.exists())
        with curator.connect(self.run / 'queue.sqlite3', True) as con:
            row = con.execute('SELECT state,error FROM jobs').fetchone()
        self.assertEqual(('skipped', 'image_changed'), tuple(row))

    def test_worker_uses_separate_safe_artifacts_for_same_uid_in_different_collections(self):
        self.add_item('../escape', collection='first')
        self.add_item('../escape', collection='second')
        self.snapshot()
        paths = []
        self.run_mock_worker(hook=lambda job, artifact: paths.append(artifact), workers=2)
        self.assertEqual(2, len(paths))
        self.assertNotEqual(paths[0], paths[1])
        for path in paths:
            path.resolve().relative_to((self.run / 'artifacts').resolve())


if __name__=='__main__':unittest.main()
