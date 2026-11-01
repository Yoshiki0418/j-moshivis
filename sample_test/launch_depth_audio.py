"""Detach a dedicated audio adaptation run from SSH/terminal lifetime."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--run-dir',type=Path,required=True)
p.add_argument('--gpu',default='1')
p.add_argument('--script',choices=['train_depth_audio.py','train_depth_audio_timing.py'],default='train_depth_audio.py')
p.add_argument('training_args',nargs=argparse.REMAINDER)
args=p.parse_args()
out=args.run_dir.resolve();out.mkdir(parents=True,exist_ok=True)
pidfile=out/'launcher.pid'
if pidfile.exists():
    pid=int(pidfile.read_text())
    try:os.kill(pid,0)
    except ProcessLookupError:pass
    else:sys.exit(f'Existing process {pid}; refusing duplicate launch')
env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,PYTHONPATH=str(ROOT/'j-moshivis'),
         NO_TORCH_COMPILE='1',TORCHDYNAMO_DISABLE='1')
extra=args.training_args
if extra[:1]==['--']:extra=extra[1:]
with (out/'training.log').open('ab',buffering=0) as log:
    process=subprocess.Popen([str(ROOT/'.venv-train/bin/python'),'-u',
        str(ROOT/'sample_test'/args.script),'--run-dir',str(out),*extra],
        cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True)
pidfile.write_text(str(process.pid))
print(f'Launched PID {process.pid}; log {out}/training.log')
