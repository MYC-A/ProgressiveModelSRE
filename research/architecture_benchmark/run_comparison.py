"""Sequential CPU stages; downloads can proceed independently, bounded waits only."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CACHE=Path('A:/ProgressiveModelSRE-cache/models')
status={}

def run(name,script,args=()):
    start=time.monotonic()
    with (ROOT/(name+'.log')).open('w',encoding='utf-8') as log:
        process=subprocess.run([sys.executable,'-u',str(ROOT/script),*args],stdout=log,stderr=subprocess.STDOUT)
    status[name]=dict(exit_code=process.returncode,seconds=time.monotonic()-start)
    (ROOT/'run_status.json').write_text(json.dumps(status,indent=2))
    print('STAGE',name,status[name],flush=True)

run('modernbert','pretrained.py',['modernbert'])
run('flan','pretrained.py',['flan_large'])
run('lamini','pretrained.py',['lamini_flan_large'])
run('train-stream','train_stream.py')
for alias in ['qwen3_06','qwen35_08']:
    start=time.monotonic()
    while True:
        registry=ROOT/'models.local.json'
        ready=registry.exists() and any(s['alias']==alias for s in json.loads(registry.read_text()))
        if ready: break
        if time.monotonic()-start>1800:
            status[alias]=dict(blocked='verified weights not ready after bounded 30 minute wait')
            break
        print('WAITING_VERIFIED_WEIGHTS',alias,round(time.monotonic()-start),flush=True)
        time.sleep(30)
    else: continue
    if ready: run(alias,'pretrained.py',[alias])
(ROOT/'run_status.json').write_text(json.dumps(status,indent=2))
print('COMPARISON_STAGES_FINISHED',flush=True)
