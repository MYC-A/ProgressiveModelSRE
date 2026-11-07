"""Check immutable inputs, complete test coverage, metrics and stream provenance."""
import hashlib
import json
from pathlib import Path
import numpy as np
from common import ROOT,load,dump,metrics,verify_inputs
from numeric import extract,execute
from stream_data import encode_window

verify_inputs()
test={r['id']:r for r in load('test')}
diagnostic={r['id']:r for r in load('diagnostic')}
stream={r['id']:r for r in load('stream/test')}
checks={}
required=['numeric_executor','flan_extract_execute','qwen35_guarded']
required += [f'{alias}_{suite}' for alias in ['flan_large','lamini_flan_large','qwen3_06','qwen35_08'] for suite in ['test','diagnostic']]
required += [f'text_{kind}_{seed}' for kind in ['mean','gru','lstm','tcn','ssd'] for seed in [0,1,2]]
required += [f'stream_{kind}_{seed}' for kind in ['mean','gru','lstm','tcn','ssd','gnn'] for seed in [0,1,2]]
required += [f'{kind}_{seed}' for kind in ['modernbert_head','modernbert_numeric','modernbert_guarded'] for seed in [0,1,2]]
missing=[name for name in required if not (ROOT/'results'/(name+'.json')).exists()]
for path in sorted((ROOT/'results').glob('*.json')):
    data=json.loads(path.read_text(encoding='utf-8'))
    rows=data['predictions']
    reference=stream if path.stem.startswith('stream_') else diagnostic if path.stem.endswith('_diagnostic') else test
    assert len(rows)==len(reference) and {r['id'] for r in rows}==set(reference),path
    for r in rows:
        original=reference[r['id']]
        for field in ['expected','suite']+(['prompt'] if reference is not stream else []):
            assert r[field]==original[field],(path,r['id'],field)
    assert metrics(rows)==data['metrics'],path
    assert data['latency']['p95_ms']>0
    checks[path.stem]=dict(n=len(rows),generation_limit=sum(r.get('hit_token_limit',False) for r in rows),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest())
protocol=load('stream/protocol')
raw=load('stream/latency_inputs')[0]
changed=[]
for line in raw:
    fields=line.split(maxsplit=9)
    fields[0]='FAKE_ALERT_LABEL'; fields[8]='FAKE_SEVERITY'
    changed.append(' '.join(fields))
assert np.array_equal(encode_window(raw),encode_window(changed)), 'Label/severity leakage'
for criterion,text in [
    ('Only result code 0 means failure; every other code means success.','Current result for job: code 0.'),
    ('Latency must be at most 4 ms; otherwise the run fails.','Current latency for job: 3 seconds.'),
    ('Only result code 1001 means success; every other code means failure.','Current result for job: code 1,000.'),
    ('Only result code 0 means success; every other code means failure.','Earlier result for job: code 0.')]:
    prompt=f'Target: job\nSuccess criterion: {criterion}\nText: {text}'
    assert execute(extract(prompt)[1])=='unknown'
assert all(execute(extract(r['prompt'])[1])==r['prediction']
           for r in load('results/numeric_executor')['predictions'])
spans=protocol['splits']
assert spans['train']['line_range'][1]==spans['validation']['line_range'][0]
assert spans['validation']['line_range'][1]==spans['test']['line_range'][0]
for split in ['train','validation','test']:
    records=load('stream/'+split)
    starts=[r['start_line_zero_based'] for r in records]
    lo,hi=spans[split]['line_range']
    assert all(lo<=s and s+32<=hi for s in starts)
    assert all(b-a>=32 for a,b in zip(starts,starts[1:]))
    data=np.load(ROOT/'stream'/(split+'.npz'))
    assert data['x'].shape==(len(records),32,129)
    assert np.isfinite(data['x']).all()
    assert sum(r['expected']=='failure' for r in records)==int(data['y'].sum())
dump(ROOT/'verification.json',dict(complete=not missing,missing=missing,result_sets=checks,
     frozen_inputs_verified=True,chronology_and_nonoverlap_verified=True,
     label_and_severity_perturbation_control_passed=True,unsupported_numeric_controls_abstain=True))
print('VERIFIED',len(checks),'sets; missing:',missing)
if missing: raise SystemExit(1)
