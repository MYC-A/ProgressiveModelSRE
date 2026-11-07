"""Freeze synthetic, criterion-conditioned SRE data before any candidate outputs."""
import hashlib
import json
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parent
INSTRUCTION=('Determine the current outcome of the target from supplied evidence and the criterion. '
             'Use unknown when evidence is absent, contradictory, about another target or only historical. '
             'Answer exactly one word: success, failure, or unknown.')
LABELS=['success','failure','unknown']

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def make(split,n,seed):
    rng=random.Random(seed)
    low,high={'train':(2,99),'validation':(100,999),'test':(1000,9999)}[split]
    rows=[]
    for family in ['code','latency','throughput','completion','recovery','binding']:
        for i in range(n):
            target=f'job_{rng.randrange(low,high)}'
            other=f'other_{rng.randrange(low,high)}'
            limit=rng.randrange(low,high)
            delta=rng.randrange(1,min(10,limit))
            for label in LABELS:
                ok=label=='success'
                if family in ('code','binding'):
                    actual=limit if ok else limit+delta
                    criteria={
                        'train':[f'Only result code {limit} means success; every other code means failure.',
                                 f'The required result code is {limit}. A different code indicates failure.'],
                        'validation':[f'A run succeeds exactly when its result code equals {limit}.'],
                        'test':[f'Treat a result as successful if and only if its code matches {limit}; otherwise it failed.',
                                f'Code {limit} is the sole successful outcome. All other result codes are failures.']}[split]
                    criterion=rng.choice(criteria)
                    observed={
                        'train':f'Current result for {target}: code {actual}.',
                        'validation':f'{target} produced result code {actual} in the current run.',
                        'test':f'The latest execution of {target} returned code {actual}.'}[split]
                elif family in ('latency','throughput'):
                    le=family=='latency'
                    actual=limit-delta if ok==le else limit+delta
                    unit='ms' if le else 'items/s'
                    if split=='train':
                        criterion=f'{family.capitalize()} must be {"at most" if le else "at least"} {limit} {unit}; otherwise the run fails.'
                        observed=f'Current {family} for {target}: {actual} {unit}.'
                    elif split=='validation':
                        criterion=f'The acceptable {family} is {"no greater" if le else "no smaller"} than {limit} {unit}.'
                        observed=f'The latest measurement for {target} reports {family} {actual} {unit}.'
                    else:
                        criterion=f'A successful run requires {family} {"not exceeding" if le else "reaching or exceeding"} {limit} {unit}.'
                        observed=f'For {target}, the current measured {family} equals {actual} {unit}.'
                elif family=='completion':
                    criterion='Success requires the requested artifact to have been created; an explicit creation failure means failure.'
                    observed={
                        'train':f'{target}: artifact {"created" if ok else "creation failed"} in the current run.',
                        'validation':f'The latest run of {target} {"successfully produced" if ok else "could not produce"} the requested artifact.',
                        'test':f'For the current execution of {target}, {"the requested output now exists" if ok else "writing the requested output was unsuccessful"}.'}[split]
                else:
                    criterion='Use the final current state. A recovered healthy service is successful; a currently unavailable service is failed.'
                    observed={
                        'train':f'{target} was unhealthy earlier. It is now {"healthy after recovery" if ok else "still unavailable"}.',
                        'validation':f'An earlier incident affected {target}. Its latest state is {"restored and healthy" if ok else "unavailable with recovery unsuccessful"}.',
                        'test':f'Previously {target} had a fault. At present, {"recovery has completed and the service is operational" if ok else "the outage persists and the service cannot be used"}.'}[split]
                if label=='unknown':
                    flavor=i%4
                    observed=[f'The current outcome of {target} has not been reported.',
                              observed.replace(target,other),
                              f'Historical record only, not evidence about the current run: {observed}',
                              f'Unresolved conflicting reports about the same current run of {target}: one says success, the other says failure.'][flavor]
                elif family=='binding':
                    distractor=observed.replace(target,other).replace(f'code {actual}',f'code {limit+delta if ok else limit}')
                    observed=distractor+' '+observed if i%2 else observed+' '+distractor
                prompt=f'{INSTRUCTION}\nTarget: {target}\nSuccess criterion: {criterion}\nText: {observed}\nAnswer:'
                rows.append(dict(id=f'{split}_{family}_{i}_{label}',suite=f'sre_{family}',family=family,
                                 prompt=prompt,expected=label,kind='state',split=split))
    rng.shuffle(rows)
    return rows

if __name__=='__main__':
    assert not (ROOT/'results').exists(), 'Do not regenerate data after observing candidate outputs'
    for split,n,seed in [('train',60,177),('validation',12,178),('test',24,179)]:
        dump(ROOT/(split+'.json'),make(split,n,seed))
    diagnostic=json.loads((ROOT.parent/'probes.frozen.json').read_text(encoding='utf-8'))
    for r in diagnostic:
        if r['kind']=='state':
            r['expected']={'OK':'success','FAIL':'failure','UNKNOWN':'unknown'}[r['expected']]
    dump(ROOT/'diagnostic.json',diagnostic)
    paths=[ROOT/(s+'.json') for s in ('train','validation','test','diagnostic')]
    dump(ROOT/'protocol.json',dict(frozen_before_candidate_outputs=True,seed=177,
         files={p.name:dict(n=len(json.loads(p.read_text())),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths},
         device='CPU',threads=4,labels=LABELS,selection='Validation only; numeric ranges and success/failure evidence wording differ by split; some criterion and unknown patterns shared',
         caveats=['Synthetic pilot, not production evidence','No full previous 120-case corpus available locally',
                  'BGL is a separate binary alert task; it has no ground-truth unknown class',
                  'Pretraining differs across ready-made models; no isolated architecture superiority claim']))
    print('FROZEN',[(p.name,json.loads(p.read_text()).__len__()) for p in paths])
