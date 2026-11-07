"""Summaries and 2x2 equality contrasts which cancel additive label/value biases."""
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def contrasts(rows):
    lookup={(r['a'],r['b']):r for r in rows if r.get('suite',r.get('group'))=='pure_equality'}
    result=[]
    for a,b in [(0,2),(200,503),(7,9)]:
        if not all(pair in lookup for pair in [(a,a),(b,b),(a,b),(b,a)]):
            continue
        def margin(x,y):
            r=lookup[x,y]
            if 'yes_minus_no' in r: return r['yes_minus_no']
            yes=sum(t['p'] for t in r['first_token_top8'] if t['id'] in (2163,4273))
            no=sum(t['p'] for t in r['first_token_top8'] if t['id'] in (465,150))
            return math.log(yes/no)
        equal=(margin(a,a)+margin(b,b))/2
        different=(margin(a,b)+margin(b,a))/2
        result.append(dict(values=[a,b],mean_equal_margin=equal,mean_different_margin=different,
                           equality_interaction=equal-different))
    return result


def summarize(rows):
    groups=sorted({r.get('suite',r.get('group')) for r in rows})
    return {g:dict(n=len(rs),correct=sum(r['correct'] for r in rs),
                   expected_yes_correct=sum(r['correct'] for r in rs if r['expected']=='yes'),
                   expected_no_correct=sum(r['correct'] for r in rs if r['expected']=='no'),
                   invalid=sum(r.get('prediction')=='INVALID' for r in rs))
            for g in groups for rs in [[r for r in rows if r.get('suite',r.get('group'))==g]]}


if __name__=='__main__':
    aggregate={}
    for alias in ['lamini_flan_large','flan_large']:
        base=ROOT/'prior_diagnosis/results'/alias/'predictions.json'
        rs=json.loads(base.read_text(encoding='utf-8'))
        aggregate['prior_'+alias]=dict(contrasts=contrasts(rs))
        for p in (ROOT/'results'/alias).glob('*.json'):
            data=json.loads(p.read_text(encoding='utf-8'))
            if isinstance(data,list) and data and 'suite' in data[0]:
                aggregate[f'{alias}/{p.stem}']=dict(summary=summarize(data),contrasts=contrasts(data))
    (ROOT/'analysis.json').write_text(json.dumps(aggregate,indent=2),encoding='utf-8')
    print(json.dumps(aggregate,indent=2))
