"""Energy-based timing diagnostic, not VAD or a correctness score.

Backchannels are allowed in full duplex: overlap alone is not a failure.
Report early text and sustained waveform energy separately for review.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf


def measure(directory):
    directory = Path(directory)
    result = json.loads((directory/'result.json').read_text())
    wav, sr = sf.read(directory/'conversation_stereo.wav', dtype='float32')
    size = round(sr * .08)
    count = len(wav)//size
    rms = np.sqrt(np.mean(wav[:count*size].reshape(count,size,2)**2, axis=1))
    active = rms > .01
    onset = np.flatnonzero(active[:,0])
    first_user = int(onset[0]) if len(onset) else count
    early = active[:first_user,1]
    overlap = active[:,0] & active[:,1]
    longest = current = 0
    for value in overlap:
        current = current+1 if value else 0
        longest = max(longest,current)
    report = {'window_sec':.08,'rms_threshold':.01,
        'first_user_energy_sec':first_user*.08,
        'assistant_active_before_first_user_sec':float(early.sum()*.08),
        'total_energy_overlap_sec':float(overlap.sum()*.08),
        'longest_continuous_energy_overlap_sec':longest*.08,
        'early_text_events':[e for e in result['raw_text_events']
                             if e['time_sec'] < first_user*.08 and e['token_id'] not in [0,3,9]],
        'note':'Energy proxy only; not speech VAD. Brief full-duplex backchannels are not errors.'}
    (directory/'turn_timing.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('directories',nargs='+');args=p.parse_args()
    for directory in args.directories:
        print(directory,json.dumps(measure(directory),ensure_ascii=False))
