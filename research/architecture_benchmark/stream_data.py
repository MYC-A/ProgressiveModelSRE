"""Chronological non-overlapping sampled windows from the original complete BGL."""
import hashlib
import json
import math
import random
import re
from pathlib import Path
import numpy as np
from common import ROOT,dump

SOURCE=Path('A:/ProgressiveModelSRE-cache/datasets/BGL.log')
WINDOW=32
DIM=129

def message_features(line,previous=None):
    fields=line.rstrip().split(maxsplit=9)
    assert len(fields)>=9,repr(line[:100])
    # Drop the ground-truth label and severity. The original message is retained.
    text=fields[9].lower() if len(fields)>9 else ''
    out=np.zeros(DIM,dtype=np.float32)
    for token in re.findall(r'[a-z]+|\d+|[^\w\s]',text):
        raw=int.from_bytes(hashlib.blake2s(token.encode(),digest_size=4).digest(),'little')
        out[raw%128]+=1 if raw&(1<<31) else -1
    out[:128]/=max(float(np.linalg.norm(out[:128])),1.)
    stamp=int(fields[1])
    out[-1]=math.log1p(max(0,stamp-previous))/15 if previous is not None else 0.
    template=re.sub(r'0x[0-9a-f]+|\b\d+\b','<N>',text)
    return out,stamp,fields[0]!='-',template

def encode_window(lines):
    features=[]; previous=None
    for line in lines:
        feature,previous,_,_=message_features(line,previous)
        features.append(feature)
    return np.stack(features)

if __name__=='__main__':
    assert not list((ROOT/'results').glob('stream_*.json')), 'Do not change data after outputs'
    n=4747963
    ranges={'train':(0,int(n*.6),3000),'validation':(int(n*.6),int(n*.8),800),
            'test':(int(n*.8),n,1000)}
    rng=random.Random(718)
    starts={}; arrays={}; labels={}; records={}; raw_timing=[]
    for split,(lo,hi,count) in ranges.items():
        candidates=list(range(lo,hi-WINDOW+1,WINDOW))
        chosen=sorted(rng.sample(candidates,count))
        starts.update({s:(split,j) for j,s in enumerate(chosen)})
        arrays[split]=np.zeros((count,WINDOW,DIM),dtype=np.float32)
        labels[split]=np.zeros(count,dtype=np.int64)
        records[split]=[dict(id=f'bgl_{s}',start_line_zero_based=s,length=WINDOW,unseen_events=0) for s in chosen]
    train_templates=set(); current=None; previous=None; window_lines=[]
    with SOURCE.open(encoding='utf-8',errors='replace') as f:
        for i,line in enumerate(f):
            if i in starts:
                current=(*starts[i],i); previous=None; window_lines=[]
            if current is None: continue
            split,j,start=current
            feature,previous,alert,template=message_features(line,previous)
            arrays[split][j,i-start]=feature
            labels[split][j]|=int(alert)
            if split=='train': train_templates.add(template)
            elif template not in train_templates: records[split][j]['unseen_events']+=1
            if split=='test' and j<30: window_lines.append(line)
            if i-start==WINDOW-1:
                if split=='test' and j<30: raw_timing.append(window_lines)
                current=None
    assert i+1==n,(i+1,n)
    folder=ROOT/'stream'; folder.mkdir(exist_ok=True)
    for split in ranges:
        np.savez_compressed(folder/(split+'.npz'),x=arrays[split],y=labels[split])
        for r,y in zip(records[split],labels[split]):
            r.update(expected='failure' if y else 'success',suite='bgl_window')
        dump(folder/(split+'.json'),records[split])
    dump(folder/'latency_inputs.json',raw_timing)
    dump(folder/'protocol.json',dict(seed=718,window=WINDOW,source=str(SOURCE),source_lines=n,
         source_sha256=hashlib.file_digest(SOURCE.open('rb'),'sha256').hexdigest(),
         splits={s:dict(line_range=[lo,hi],windows=count,alerts=int(labels[s].sum())) for s,(lo,hi,count) in ranges.items()},
         labels='window is anomalous if any original alert label occurs inside',
         features='signed 128-dimensional message token hashing + timestamp gap; labels and severity excluded',
         limitations=['Chronological sampled-window pilot, not full-corpus performance',
                      'Templates may recur across chronological splits; unseen-event counts recorded',
                      'Global log order mixes nodes; no service-causality claim','BGL has no unknown labels']))
    print('STREAM_FROZEN',[(s,len(labels[s]),int(labels[s].sum())) for s in ranges],flush=True)
