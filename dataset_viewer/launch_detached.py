"""Launch the CPU-only dataset viewer independently of an SSH session."""
import os
import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
with socket.socket() as probe:
    if probe.connect_ex(('127.0.0.1', 8765)) == 0:
        raise SystemExit('Port 8765 already in use; existing process left untouched')
logs = ROOT / 'logs'
logs.mkdir(exist_ok=True)
with (logs / 'dataset_viewer_20261109.log').open('ab', buffering=0) as log:
    process = subprocess.Popen(
        [str(ROOT / '.venv-data/bin/python'), '-u', '-m', 'dataset_viewer.app_daily'],
        cwd=ROOT, env=dict(os.environ), stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
(logs / 'dataset_viewer_20261109.pid').write_text(str(process.pid))
print(f'Started PID {process.pid}: http://127.0.0.1:8765')
