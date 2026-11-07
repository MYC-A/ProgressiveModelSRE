"""Audit frozen inputs, full-generation evidence, and weight-restoration controls."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
probes_path = ROOT/'probes.frozen.json'
protocol = json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
assert hashlib.sha256(probes_path.read_bytes()).hexdigest() == protocol['probes_sha256']
probes = {r['id']: r for r in json.loads(probes_path.read_text(encoding='utf-8'))}
assert len(probes) == 122
audit = {}
for alias in ('lamini_flan_large', 'flan_large'):
    folder = ROOT/'results'/alias
    baseline = {r['id']: r for r in json.loads((folder/'baseline.json').read_text())}
    required = ['baseline', 'swap_dense_all', 'swap_norm_all', 'swap_dense_21_23', 'ff_skip_21_23']
    for name in required:
        assert (folder/(name+'.json')).exists(), (alias, name)
    for path in sorted(folder.glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, list) or not data or 'suite' not in data[0]:
            continue
        ids = [r['id'] for r in data]
        assert len(ids) == len(set(ids)), path
        if path.stem != 'restoration_control':
            assert set(ids) == set(probes), path
        for r in data:
            assert r['complete_generation'], (path, r['id'])
            for key in ('prompt', 'expected', 'suite'):
                assert r[key] == probes[r['id']][key], (path, r['id'], key)
            assert r['input_token_ids'] == baseline[r['id']]['input_token_ids']
            assert all(math.isfinite(r[k]) for k in ('yes_minus_no', 'p_yes', 'p_no'))
        audit[f'{alias}/{path.stem}'] = dict(n=len(data),
            invalid=sum(r['prediction']=='INVALID' for r in data),
            generation_limit=sum(r['hit_token_limit'] for r in data))
    restored = json.loads((folder/'restoration_control.json').read_text())
    assert len(restored) == 12
    assert all(r['output_text'] == baseline[r['id']]['output_text'] for r in restored)
    delta = max(abs(r['yes_minus_no']-baseline[r['id']]['yes_minus_no']) for r in restored)
    assert delta < 1e-4
    audit[f'{alias}/restoration_control']['max_margin_difference'] = delta
(ROOT/'evidence_verification.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print('VERIFIED_EVIDENCE', len(audit), 'result sets; frozen prompts and restoration controls passed')
