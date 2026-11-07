import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
summary={}
for alias in ('lamini_flan_large','flan_large'):
    path=ROOT/'results'/alias/'baseline.json'
    if not path.exists(): continue
    current=json.loads(path.read_text(encoding='utf-8'))
    prior={r['id']:r for r in json.loads((ROOT/'prior_diagnosis/results'/alias/'predictions.json').read_text(encoding='utf-8'))}
    rows=[]
    for r in current:
        if r['suite'] not in ('pure_equality','original'): continue
        old=prior[r['id']]
        old_probs={t['id']:t['p'] for t in old['first_token_top8']}
        common=[abs(t['p']-old_probs[t['id']]) for t in r['top6'] if t['id'] in old_probs]
        rows.append(dict(id=r['id'],same_complete_output=r['output_text']==old['output_text'],
                         max_probability_difference=max(common)))
    assert all(r['same_complete_output'] for r in rows),alias
    assert max(r['max_probability_difference'] for r in rows)<1e-4,alias
    summary[alias]=dict(n=len(rows),all_complete_outputs_match=True,
                        max_probability_difference=max(r['max_probability_difference'] for r in rows),cases=rows)
(ROOT/'reproduction_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print({k:{n:v for n,v in d.items() if n!='cases'} for k,d in summary.items()})
