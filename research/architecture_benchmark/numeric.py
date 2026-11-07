"""Explicit schema-limited numeric feature extractor; no gold fields are read."""
import re
import numpy as np

def extract(prompt):
    fields={k:v for k,v in re.findall(r'^(Target|Success criterion|Text): (.*)$',prompt,re.M)}
    target=fields.get('Target',''); criterion=fields.get('Success criterion','').lower()
    text=fields.get('Text',''); vector=np.zeros(8,dtype=np.float32)
    record=dict(valid=False,reason='unsupported numeric schema')
    if not target or not re.search(r'code|latency|throughput',criterion): return vector,record
    if ('Historical record only' in text or 'Unresolved conflicting reports' in text
            or 'has not been reported' in text):
        return vector,dict(valid=False,reason='absent, historical or contradictory evidence')
    evidence=[s for s in re.split(r'\.(?!\d)',text) if re.search(r'(?<!\w)'+re.escape(target)+r'(?!\w)',s)]
    if len(evidence)!=1: return vector,dict(valid=False,reason='target evidence not unique')
    if not re.search(r'\b(current|latest)\b',evidence[0],re.I):
        return vector,dict(valid=False,reason='current observation not established by supported schema')
    number=r'(-?\d+(?:\.\d+)?)'
    name=re.escape(target)
    if 'latency' in criterion or 'throughput' in criterion:
        term='latency' if 'latency' in criterion else 'throughput'
        unit='ms' if term=='latency' else 'items/s'
        op='le' if term=='latency' else 'ge'
        phrases=['at most','no greater than','not exceeding'] if op=='le' else ['at least','no smaller than','reaching or exceeding']
        criterion_patterns=[rf'{term} must be {phrases[0]} {number} {unit}; otherwise the run fails\.',
            rf'the acceptable {term} is {phrases[1]} {number} {unit}\.',
            rf'a successful run requires {term} {phrases[2]} {number} {unit}\.']
        observed_patterns=[rf'current {term} for {name}: {number} {unit}',
            rf'the latest measurement for {name} reports {term} {number} {unit}',
            rf'for {name}, the current measured {term} equals {number} {unit}']
    else:
        op='eq'
        criterion_patterns=[rf'only result code {number} means success; every other code means failure\.',
            rf'the required result code is {number}\. a different code indicates failure\.',
            rf'a run succeeds exactly when its result code equals {number}\.',
            rf'treat a result as successful if and only if its code matches {number}; otherwise it failed\.',
            rf'code {number} is the sole successful outcome\. all other result codes are failures\.']
        observed_patterns=[rf'current result for {name}: code {number}',
            rf'{name} produced result code {number} in the current run',
            rf'the latest execution of {name} returned code {number}']
    def match_one(patterns,text):
        return next((m for p in patterns if (m:=re.fullmatch(p,text.strip(),re.I))),None)
    observed=match_one(observed_patterns,evidence[0])
    expected=match_one(criterion_patterns,criterion)
    if not observed or not expected: return vector,dict(valid=False,reason='unsupported observation, units or operator grammar')
    a,e=float(observed[1]),float(expected[1])
    vector[:4]=[1,float(a==e),float(a<e),float(a>e)]
    vector[4+['eq','le','ge'].index(op)]=1
    vector[7]=np.tanh((a-e)/max(abs(a),abs(e),1))
    return vector,dict(valid=True,actual=a,expected=e,operator=op,evidence=evidence[0].strip(),criterion=criterion)

def execute(record):
    if not record['valid']: return 'unknown'
    a,e,op=record['actual'],record['expected'],record['operator']
    ok={'eq':a==e,'le':a<=e,'ge':a>=e}[op]
    return 'success' if ok else 'failure'
