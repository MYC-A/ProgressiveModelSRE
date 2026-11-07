"""Run stages as their verified checkpoint files become available."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent


def wait_for(alias):
    while True:
        path = root/'models.local.json'
        if path.exists():
            try:
                if any(s['alias']==alias for s in json.loads(path.read_text())):
                    return
            except json.JSONDecodeError:
                pass
        print('WAITING_FOR_VERIFIED_CHECKPOINT', alias, flush=True)
        time.sleep(30)


def run(alias, stage):
    subprocess.run([sys.executable, '-u', str(root/'investigate.py'), '--model', alias, '--stage', stage], check=True)


wait_for('lamini_flan_large')
run('lamini_flan_large', 'baseline')
run('lamini_flan_large', 'mechanism')
run('lamini_flan_large', 'heads')
run('lamini_flan_large', 'positions')
wait_for('flan_large')
run('flan_large', 'baseline')
subprocess.run([sys.executable, '-u', str(root/'extract_donors.py')], check=True)
run('lamini_flan_large', 'hybrid')
run('flan_large', 'hybrid')
run('flan_large', 'controls')
for script in ['verify_reproduction.py','verify_results.py','analyze_activations.py','build_report.py','plot_results.py','package_results.py']:
    subprocess.run([sys.executable,str(root/script)],check=True)
print('ALL_STAGES_COMPLETED', flush=True)
