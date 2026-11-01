"""Start the local-only realtime server detached from SSH. Never kill other jobs."""
import os
import socket
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
with socket.socket() as probe:
    if probe.connect_ex(('127.0.0.1',8998))==0:
        raise SystemExit('Port 8998 already in use; existing server left untouched')
logs=ROOT/'logs';logs.mkdir(exist_ok=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1')
with (logs/'realtime_web_20261015.log').open('ab',buffering=0) as log:
    process=subprocess.Popen(['bash',str(ROOT/'run_web_demo.sh')],cwd=ROOT,
        env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(logs/'realtime_web_20261015.pid').write_text(str(process.pid))
print('Started PID',process.pid,'on GPU1; log',logs/'realtime_web_20261015.log')
