"""Content-bound, resumable Codex CLI pre-correction; never approves data.

Run `snapshot` once, then `run` with a bounded number of workers. Suggestions
are separate from source data and human review history. No API key is used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time

from review_store import ReviewStore, content_fingerprint, image_file_sha256
from ai_suggestion_store import human_touched_uids

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'data/curation/codex_luna_pending_20261117'
CATALOG = ROOT / 'data/.dataset_viewer/catalog.sqlite3'
PROMPT = Path(__file__).with_name('curation_prompt_ja.md')
STOP = threading.Event()
SCHEMA = {
    'type':'object','additionalProperties':False,
    'properties':{
        'turns':{'type':'array','items':{'type':'object','additionalProperties':False,
            'properties':{'turn_index':{'type':'integer'},'text':{'type':'string'}},'required':['turn_index','text']}},
        'auxiliary_update':{'type':'object','additionalProperties':False,'properties':{
            'answer_count':{'type':['integer','null']},'counted_target':{'type':['string','null']},
            'recognized_text':{'type':['string','null']}},'required':['answer_count','counted_target','recognized_text']},
        'changes':{'type':'array','items':{'type':'object','additionalProperties':False,
            'properties':{'turn_index':{'type':'integer'},'reason':{'type':'string'}},'required':['turn_index','reason']}},
        'requires_attention':{'type':'boolean'},'concerns':{'type':'array','items':{'type':'string'}}},
    'required':['turns','auxiliary_update','changes','requires_attention','concerns']}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


@contextmanager
def connect(path, readonly=False):
    c = sqlite3.connect(f'file:{path}?mode=ro',uri=True,timeout=30) if readonly else sqlite3.connect(path,timeout=30)
    c.row_factory=sqlite3.Row
    try:
        with c:
            yield c
    finally:
        c.close()


def read_record(row, sources):
    p=ROOT / sources[row['dialogue_source']]
    with p.open('rb') as f:
        f.seek(row['dialogue_offset']); raw=f.read(row['dialogue_length'])
    return json.loads(raw)


def touched(store):
    return human_touched_uids(store)


def artifact_key(job):
    """Opaque, path-safe identity including collection and source revision."""
    identity = [job[key] for key in ('collection', 'uid', 'base_fingerprint', 'image_sha256')]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def sources_map(c):
    return {r['source_id']:r['path'] for r in c.execute('select source_id,path from sources')}


def snapshot(run):
    run.mkdir(parents=True,exist_ok=True)
    if (run/'queue.sqlite3').exists():raise RuntimeError('Snapshot already exists; resume run instead')
    reviews=ReviewStore(ROOT/'data/curation/dialogue_review_events.jsonl')
    human=touched(reviews)
    with connect(CATALOG,True) as cat,connect(run/'queue.sqlite3') as q:
        q.execute('PRAGMA journal_mode=WAL')
        q.execute('CREATE TABLE jobs (collection TEXT,uid TEXT,base_fingerprint TEXT,image_sha256 TEXT,record_json TEXT,image_path TEXT,state TEXT DEFAULT \'pending\',attempts INTEGER DEFAULT 0,error TEXT,updated_at TEXT, PRIMARY KEY(collection,uid))')
        src=sources_map(cat)
        n=0
        for r in cat.execute("SELECT * FROM items WHERE review_required=1 AND review_status='pending' AND has_dialogue=1 AND has_image=1 ORDER BY collection,category,uid"):
            if (r['collection'],r['uid']) in human:continue
            summary=json.loads(r['summary_json'])
            if summary.get('human_revision') or summary.get('image_integrity_status')!='verified':continue
            record=read_record(r,src)
            image_hash=summary.get('current_image_sha256')
            if content_fingerprint(record,image_sha256=image_hash)!=r['review_id']:continue
            q.execute('INSERT INTO jobs(collection,uid,base_fingerprint,image_sha256,record_json,image_path,updated_at) VALUES(?,?,?,?,?,?,?)',
                (r['collection'],r['uid'],r['review_id'],image_hash,json.dumps(record,ensure_ascii=False),r['image_path'],utc()))
            n+=1
    (run/'snapshot.json').write_text(json.dumps({'created_at':utc(),'jobs':n,'selector':'required + pending + image + dialogue; any human event excluded','model':'gpt-6-luna'},ensure_ascii=False,indent=2))
    print('snapshot jobs',n,flush=True)


def eligible(job):
    """Recheck human decisions, current source bytes, and image at both ends."""
    reviews=ReviewStore(ROOT/'data/curation/dialogue_review_events.jsonl')
    if (job['collection'],job['uid']) in touched(reviews):return False,'human_review_exists'
    with connect(CATALOG,True) as cat:
        r=cat.execute('select * from items where collection=? and uid=?',(job['collection'],job['uid'])).fetchone()
        if not r or not r['review_required'] or r['review_status']!='pending':return False,'no_longer_pending'
        if r['review_id']!=job['base_fingerprint']:return False,'source_fingerprint_changed'
        record=read_record(r,sources_map(cat))
        image=ROOT/r['image_path']
        if image_file_sha256(image)!=job['image_sha256']:return False,'image_changed'
        if content_fingerprint(record,image_sha256=job['image_sha256'])!=job['base_fingerprint']:return False,'source_content_changed'
        if r['image_path']!=job['image_path']:return False,'image_path_changed'
    return True,None


def validate(response,record):
    if set(response)!=set(SCHEMA['required']):raise ValueError('Unexpected response keys')
    old=record['dialogue']; turns=response['turns']
    if len(turns)!=len(old):raise ValueError('Turn count changed')
    dialogue=[]
    for i,(a,b) in enumerate(zip(old,turns)):
        if b.get('turn_index')!=i or not isinstance(b.get('text'),str) or not b['text'].strip() or len(b['text'])>1000:
            raise ValueError('Invalid turn index/text')
        dialogue.append({'speaker':a['speaker'],'text':b['text'].strip()})
    aux=json.loads(json.dumps(record.get('auxiliary_task')))
    update=response['auxiliary_update']
    if aux and aux.get('kind')=='counting':
        if update.get('recognized_text') is not None:raise ValueError('Wrong auxiliary type')
        n=update.get('answer_count')
        if n is not None:
            if type(n)!=int or not 1<=n<=1000:raise ValueError('Invalid count')
            aux['answer_count']=n
        value=update.get('counted_target')
        if value is not None:
            if not isinstance(value,str) or not value.strip() or len(value)>1000:raise ValueError('Invalid target')
            aux['counted_target']=value.strip()
    elif aux and aux.get('kind')=='ocr':
        if update.get('answer_count') is not None or update.get('counted_target') is not None:raise ValueError('Wrong auxiliary type')
        value=update.get('recognized_text')
        if value is not None:
            if not isinstance(value,str) or not value.strip() or len(value)>1000:raise ValueError('Invalid OCR text')
            aux['recognized_text']=value.strip()
    elif any(v is not None for v in update.values()):raise ValueError('Unexpected auxiliary edits')
    concerns=response['concerns']
    if type(response['requires_attention'])!=bool or not isinstance(concerns,list) or any(not isinstance(x,str) or len(x)>1500 for x in concerns):
        raise ValueError('Invalid concerns')
    changes=[]
    reasons={x['turn_index']:x['reason'] for x in response['changes'] if isinstance(x,dict) and type(x.get('turn_index'))==int and isinstance(x.get('reason'),str)}
    for i,(a,b) in enumerate(zip(old,dialogue)):
        if a['text']!=b['text']:changes.append({'turn_index':i,'reason':reasons.get(i,'画像・前後の対話との整合性を修正')[:1500]})
    changed=bool(changes) or aux!=record.get('auxiliary_task')
    attention=response['requires_attention'] or bool(concerns)
    if aux and (aux.get('kind')=='counting' and update.get('answer_count') is None or aux.get('kind')=='ocr' and update.get('recognized_text') is None):
        attention=True
        if not concerns:concerns=['補助タスクの正解を確定できていません。画像と照合してください。']
    return dict(dialogue=dialogue,auxiliary_task=aux,changes=changes,concerns=concerns,
                verdict='needs_attention' if attention else ('corrected' if changed else 'unchanged'))


def invoke_codex(executable,job,artifact,timeout):
    artifact.mkdir(parents=True,exist_ok=True)
    (artifact/'identity.json').write_text(json.dumps({key:job[key] for key in (
        'collection','uid','base_fingerprint','image_sha256')},ensure_ascii=False,indent=2))
    record=json.loads(job['record_json'])
    # Do not expose source URLs, titles, filesystem paths or unrelated records to
    # the judging model. The image is supplied through CLI's native image input.
    data={'dialogue':record['dialogue'],'auxiliary_task':record.get('auxiliary_task')}
    prompt=PROMPT.read_text()+'\n\n検査対象JSON:\n'+json.dumps(data,ensure_ascii=False)
    (artifact/'prompt.txt').write_text(prompt)
    schema=artifact/'schema.json';schema.write_text(json.dumps(SCHEMA))
    response=artifact/'response.json'
    env=dict(os.environ)
    for key in ['OPENAI_API_KEY','CODEX_API_KEY','OPENAI_BASE_URL','OPENAI_ORG_ID','OPENAI_ORGANIZATION','CODEX_THREAD_ID']:
        env.pop(key,None)
    with tempfile.TemporaryDirectory(prefix='codex-visual-curation-') as work:
        cmd=[executable,'exec','--ignore-user-config','--ephemeral','--sandbox','read-only',
             '--skip-git-repo-check','--color','never','--json','--model','gpt-6-luna','--cd',work,
             '--output-schema',str(schema),'--output-last-message',str(response),'--image',str(ROOT/job['image_path'])]
        for c in ['approval_policy="never"','web_search="disabled"','project_doc_max_bytes=0',
                  'model_provider="openai"','model_reasoning_effort="medium"','history.persistence="none"']:
            cmd+=['-c',c]
        for feature in ['shell_tool','shell_snapshot','apps','plugins','hooks','multi_agent','multi_agent_v2',
                        'browser_use','browser_use_external','computer_use','image_generation','view_image',
                        'daemon_auto_start','unbounded_connection_retries','skill_mcp_dependency_install','skill_search','sleep_tool']:
            cmd+=['--disable',feature]
        cmd+=['-']
        start=time.monotonic()
        with (artifact/'events.jsonl').open('w') as stdout,(artifact/'stderr.log').open('w') as stderr:
            try:
                proc=subprocess.run(cmd,input=prompt,text=True,stdout=stdout,stderr=stderr,env=env,timeout=timeout)
            except subprocess.TimeoutExpired as exc:raise RuntimeError('CLI timeout') from exc
        usage={};completed=0
        for line in (artifact/'events.jsonl').read_text().splitlines():
            try:e=json.loads(line)
            except json.JSONDecodeError:raise RuntimeError('CLI emitted invalid JSON event')
            if e.get('type') in ['error','turn.failed']:raise RuntimeError('CLI failed: '+json.dumps(e,ensure_ascii=False)[:800])
            if e.get('model') and e['model']!='gpt-6-luna':raise RuntimeError('Unexpected model')
            item=e.get('item')
            if item and item.get('type') not in ['reasoning','agent_message']:raise RuntimeError('Unexpected CLI tool activity')
            if e.get('type')=='turn.completed':completed+=1;usage=e.get('usage',{})
        if proc.returncode or completed!=1 or not response.is_file():
            raise RuntimeError('CLI did not complete: '+(artifact/'stderr.log').read_text()[-900:])
        return json.loads(response.read_text()),dict(model='gpt-6-luna',elapsed_seconds=round(time.monotonic()-start,2),usage=usage,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),response_sha256=hashlib.sha256(response.read_bytes()).hexdigest())


def finish(run,job,state,error=None):
    with connect(run/'queue.sqlite3') as q:
        q.execute('update jobs set state=?,error=?,updated_at=? where collection=? and uid=?',
                  (state,error,utc(),job['collection'],job['uid']))


@contextmanager
def worker_lock(run):
    with (run/'worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Worker already running')
        try:
            yield
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


def run_jobs(run,executable,workers,limit,timeout):
    with worker_lock(run):
        _run_jobs(run,executable,workers,limit,timeout)


def _run_jobs(run,executable,workers,limit,timeout):
    from ai_suggestion_store import default_ai_suggestion_store
    store=default_ai_suggestion_store(ROOT)
    with connect(run/'queue.sqlite3') as q:q.execute("update jobs set state='pending' where state='running'")
    claimed=0;claim_lock=threading.Lock()
    def work():
        nonlocal claimed
        while not STOP.is_set():
            with claim_lock,connect(run/'queue.sqlite3') as q:
                if limit and claimed>=limit:return
                q.execute('BEGIN IMMEDIATE')
                row=q.execute("select * from jobs where state='pending' order by collection,uid limit 1").fetchone()
                if row is None:return
                job=dict(row);claimed+=1
                q.execute("update jobs set state='running',attempts=attempts+1,updated_at=? where collection=? and uid=?",(utc(),job['collection'],job['uid']))
            try:
                ok,why=eligible(job)
                if not ok:finish(run,job,'skipped',why);continue
                prior=store.get_suggestion(collection=job['collection'],uid=job['uid'],base_fingerprint=job['base_fingerprint'],image_sha256=job['image_sha256'])
                if prior:finish(run,job,'done');continue
                artifact=run/'artifacts'/artifact_key(job)/f"attempt_{job['attempts']+1}"
                response,provenance=invoke_codex(executable,job,artifact,timeout)
                correction=validate(response,json.loads(job['record_json']))
                ok,why=eligible(job)
                if not ok:finish(run,job,'skipped',why);continue
                store.put_suggestion(collection=job['collection'],uid=job['uid'],base_fingerprint=job['base_fingerprint'],
                    image_sha256=job['image_sha256'],model='gpt-6-luna',provenance=provenance,**correction)
                finish(run,job,'done')
                print(utc(),job['uid'],correction['verdict'],len(correction['changes']),'turns changed',provenance['elapsed_seconds'],'sec',flush=True)
            except Exception as exc:
                message=f'{type(exc).__name__}: {exc}'[:1500]
                print(utc(),job['uid'],'ERROR',message,flush=True)
                fatal=any(s in message.lower() for s in ['usage limit','usage_limit','rate limit','rate_limit','quota','unauthorized','authentication','not supported','unknown feature','not found','unexpected cli tool','unexpected model'])
                if fatal:
                    finish(run,job,'error',message)
                    STOP.set()  # No runaway retry on quota/auth/model issues.
                elif job['attempts']+1<3:
                    finish(run,job,'pending',message)
                    STOP.wait(10)
                else:
                    finish(run,job,'error',message)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(work) for _ in range(workers)]:future.result()
    status(run)


def status(run):
    with connect(run/'queue.sqlite3',True) as q:
        print(json.dumps({r['state']:r['n'] for r in q.execute('select state,count(*) n from jobs group by state')},ensure_ascii=False),flush=True)


def resolve_codex(executable=None):
    """Resolve once before starting workers, without a machine-specific path."""
    requested = os.path.expanduser(executable) if executable else 'codex'
    found = shutil.which(requested)
    if found is None:
        raise FileNotFoundError(
            f'Codex CLI executable not found: {requested}. '
            'Add codex to PATH or pass --codex /path/to/codex.')
    return str(Path(found).resolve())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['snapshot','run','status','retry-errors'])
    p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    p.add_argument('--codex',help='Codex executable path (default: codex on PATH)')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--timeout',type=int,default=240)
    args=p.parse_args();run=args.run_dir.resolve()
    if not 1<=args.workers<=4:p.error('workers must be 1..4')
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:STOP.set())
    if args.action=='snapshot':snapshot(run)
    elif args.action=='run':
        try:executable=resolve_codex(args.codex)
        except FileNotFoundError as error:p.error(str(error))
        run_jobs(run,executable,args.workers,args.limit,args.timeout)
    elif args.action=='status':status(run)
    else:
        with connect(run/'queue.sqlite3') as q:q.execute("update jobs set state='pending',error=NULL where state='error'")
        status(run)


if __name__=='__main__':main()
