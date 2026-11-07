"""Matched-data CPU pilot: tiny non-pretrained networks, selected on validation only."""
import copy
import random
import time
import torch
from torch.nn import functional as F
from common import ROOT,LABELS,load,dump,metrics,latency,environment,verify_inputs
from networks import Classifier,batch_encode

def main():
    verify_inputs()
    train,val,test=(load(s) for s in ['train','validation','test'])
    encoded={s:batch_encode(rs) for s,rs in [('train',train),('validation',val),('test',test)]}
    labels={s:torch.tensor([LABELS.index(r['expected']) for r in rs]) for s,rs in [('train',train),('validation',val),('test',test)]}
    for kind in ['mean','gru','lstm','tcn','ssd']:
        for seed in [0,1,2]:
            out=ROOT/'results'/f'text_{kind}_{seed}.json'
            if out.exists(): continue
            torch.manual_seed(seed); random.seed(seed)
            model=Classifier(kind); optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
            best=-1; stale=0; history=[]; start=time.monotonic()
            for epoch in range(12):
                model.train()
                for ids in torch.randperm(len(train)).split(32):
                    optimizer.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(model(encoded['train'][ids]),labels['train'][ids])
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optimizer.step()
                model.eval()
                with torch.inference_mode():
                    logits=torch.cat([model(x) for x in encoded['validation'].split(32)])
                    score=float((logits.argmax(-1)==labels['validation']).float().mean())
                history.append(dict(epoch=epoch+1,validation_accuracy=score))
                print(kind,seed,history[-1],round(time.monotonic()-start,1),flush=True)
                if score>best:
                    best=score; state=copy.deepcopy(model.state_dict()); stale=0
                else: stale+=1
                if stale>=3: break
            model.load_state_dict(state); model.eval()
            with torch.inference_mode():
                predictions=torch.cat([model(x).argmax(-1) for x in encoded['test'].split(32)]).tolist()
                rows=[dict(**r,prediction=LABELS[p]) for r,p in zip(test,predictions)]
                def infer(r): return LABELS[int(model(batch_encode([r])).argmax(-1))]
                timing=latency(infer,test)
            checkpoint=ROOT/'checkpoints'/f'text_{kind}_{seed}.pt'; checkpoint.parent.mkdir(exist_ok=True)
            torch.save(state,checkpoint)
            dump(out,dict(model=kind,seed=seed,training='from scratch, synthetic train only',
                 parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-start,
                 history=history,metrics=metrics(rows),latency=timing,predictions=rows,environment=environment(),
                 implementation='dense reference SSD, not optimized Mamba-2' if kind=='ssd' else kind))
            print('SAVED',out,metrics(rows)['all'],flush=True)

if __name__=='__main__': main()
