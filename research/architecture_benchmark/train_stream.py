"""Same BGL windows/features for small supervised sequence and graph prototypes."""
import copy
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from common import ROOT,dump,metrics,latency,environment
from networks import Classifier
from stream_data import encode_window,DIM

def main():
    folder=ROOT/'stream'; data={s:np.load(folder/(s+'.npz')) for s in ['train','validation','test']}
    xs={s:torch.from_numpy(d['x']) for s,d in data.items()}
    ys={s:torch.from_numpy(d['y']) for s,d in data.items()}
    records=json.loads((folder/'test.json').read_text())
    raw=json.loads((folder/'latency_inputs.json').read_text())
    counts=torch.bincount(ys['train'],minlength=2).float()
    weights=counts.sum()/(2*counts.clamp_min(1))
    for kind in ['mean','gru','lstm','tcn','ssd','gnn']:
        for seed in [0,1,2]:
            out=ROOT/'results'/f'stream_{kind}_{seed}.json'
            if out.exists(): continue
            torch.manual_seed(seed)
            model=Classifier(kind,d=32,input_dim=DIM,classes=2)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
            best=float('inf'); history=[]; stale=0; start=time.monotonic()
            for epoch in range(12):
                model.train()
                for indices in torch.randperm(len(ys['train'])).split(64):
                    optimizer.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(model(xs['train'][indices]),ys['train'][indices],weight=weights)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optimizer.step()
                model.eval()
                with torch.inference_mode():
                    logits=torch.cat([model(x) for x in xs['validation'].split(64)])
                    loss=float(F.cross_entropy(logits,ys['validation'],weight=weights))
                history.append(dict(epoch=epoch+1,validation_loss=loss))
                print(kind,seed,history[-1],round(time.monotonic()-start,1),flush=True)
                if loss<best-1e-5:
                    best=loss; state=copy.deepcopy(model.state_dict()); stale=0
                else: stale+=1
                if stale>=3: break
            model.load_state_dict(state); model.eval()
            with torch.inference_mode():
                valp=torch.cat([model(x).softmax(-1)[:,1] for x in xs['validation'].split(64)]).numpy()
                normal=valp[ys['validation'].numpy()==0]
                threshold=float(np.quantile(normal,.99,method='higher'))
                def predict(x): return torch.cat([model(b).softmax(-1)[:,1] for b in x.split(64)]).numpy()
                p=predict(xs['test'])
                rows=[dict(**r,score=float(v),prediction='failure' if v>threshold else 'success') for r,v in zip(records,p)]
                def infer(lines):
                    x=torch.from_numpy(encode_window(lines))[None]
                    return float(model(x).softmax(-1)[0,1])>threshold
                timing=latency(infer,raw)
                rng=torch.Generator().manual_seed(991)
                order=torch.randperm(xs['test'].shape[1],generator=rng)
                shuffled=predict(xs['test'][:,order])
                shuffled_rows=[dict(**r,prediction='failure' if v>threshold else 'success') for r,v in zip(records,shuffled)]
            checkpoint=ROOT/'checkpoints'/f'stream_{kind}_{seed}.pt'; checkpoint.parent.mkdir(exist_ok=True)
            torch.save(state,checkpoint)
            dump(out,dict(model=kind,seed=seed,task='BGL chronological window alert classification',
                parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-start,
                history=history,threshold=threshold,threshold_selection='99th percentile of validation normal scores; strict >',
                metrics=metrics(rows),shuffled_metrics=metrics(shuffled_rows),latency=timing,predictions=rows,
                environment=environment(),implementation='dense SSD reference, not optimized Mamba-2' if kind=='ssd' else kind))
            print('SAVED',out,metrics(rows)['all'],flush=True)

if __name__=='__main__': main()
