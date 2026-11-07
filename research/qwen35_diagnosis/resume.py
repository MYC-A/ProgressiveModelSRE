"""Bounded, memory-conscious phase runner; completed results are never regenerated."""
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
start = time.monotonic()
amendment = dict(reason='Original FP32 batch-8 process exhausted available desktop RAM during validation before saving any validation outputs.',
    default_batch_size=2, thinking_batch_size=1, dtype='float32 unchanged',
    execution='One result file per fresh model process; resume saved phases without rerunning them.',
    initial_saved_files=sorted(p.name for p in ROOT.glob('*.json')),
    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
(ROOT / 'execution_amendment.json').write_text(json.dumps(amendment, indent=2), encoding='utf-8')
for phase in range(24):
    if (ROOT / 'completion.json').exists():
        break
    with (ROOT / f'phase_{phase:02d}.log').open('w', encoding='utf-8') as stdout, (ROOT / f'phase_{phase:02d}.err.log').open('w', encoding='utf-8') as stderr:
        status = subprocess.run([sys.executable, '-u', str(ROOT / 'run.py'), '--one-phase'], stdout=stdout, stderr=stderr).returncode
    print('PHASE', phase, 'exit', status, 'elapsed', round(time.monotonic() - start, 1), flush=True)
    if status != 0:
        raise SystemExit(status)
else:
    raise RuntimeError('Bounded phase count exceeded')
if not (ROOT / 'completion.json').exists():
    raise RuntimeError('Missing completion')
subprocess.run([sys.executable, str(ROOT / 'analyze.py')], check=True)
print('ALL COMPLETE', round(time.monotonic() - start, 1), flush=True)
