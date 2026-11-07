import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
CACHE=Path('A:/ProgressiveModelSRE-cache/models')
LABELS=['success','failure','unknown']
torch.set_num_threads(4)

def load(name):
    return json.loads((ROOT/(name+'.json')).read_text(encoding='utf-8'))

def dump(path,obj):
    path=Path(path); path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def verify_inputs():
    for name,meta in load('protocol')['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==meta['sha256'],name

def metrics(rows):
    groups={}
    for suite in sorted({'all'}|{r['suite'] for r in rows}):
        rs=[r for r in rows if suite=='all' or r['suite']==suite]
        n=len(rs); correct=sum(r['prediction']==r['expected'] for r in rs)
        good=[r for r in rs if r['expected']=='success']
        bad=[r for r in rs if r['expected']=='failure']
        unknown=[r for r in rs if r['expected']=='unknown']
        groups[suite]=dict(n=n,correct=correct,accuracy=correct/n,
            success_n=len(good),false_alarm=sum(r['prediction']=='failure' for r in good),
            failure_n=len(bad),missed_failure=sum(r['prediction']!='failure' for r in bad),
            unknown_n=len(unknown),unknown_correct=sum(r['prediction']=='unknown' for r in unknown),
            invalid=sum(r['prediction']=='INVALID' for r in rs),
            predictions=dict(Counter(r['prediction'] for r in rs)))
    return groups

def latency(fn,items,n=30):
    for item in items[:3]: fn(item)
    times=[]
    for i in range(n):
        start=time.perf_counter(); fn(items[i%len(items)])
        times.append((time.perf_counter()-start)*1000)
    return dict(n=n,p50_ms=float(np.percentile(times,50)),p95_ms=float(np.percentile(times,95)),
                includes='single item, tokenization/features + inference + parsing; warm model',
                raw_ms=times)

def environment():
    return dict(python=platform.python_version(),torch=torch.__version__,device='CPU',threads=4,
                processor=platform.processor(),warning='Shared desktop; not an isolated production server')
