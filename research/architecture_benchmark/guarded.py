"""Explicit numeric executor with a learned ModernBERT fallback; schema-limited."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0')
import re
import torch
from torch import nn
from transformers import AutoTokenizer,AutoModel
from common import ROOT,CACHE,LABELS,load,dump,metrics,latency,environment
from numeric import extract,execute

data=load('test')
tok=AutoTokenizer.from_pretrained(CACHE/'modernbert',local_files_only=True)
encoder=AutoModel.from_pretrained(CACHE/'modernbert',local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
def numeric_case(row):
    criterion=re.search(r'^Success criterion: (.*)$',row['prompt'],re.M)
    return criterion and bool(re.search(r'code|latency|throughput',criterion[1].lower()))
for seed in [0,1,2]:
    baseline=load(f'results/modernbert_head_{seed}')
    rows=[]
    for r in baseline['predictions']:
        row=dict(r)
        if numeric_case(row):
            _,record=extract(row['prompt']); row.update(prediction=execute(record),extraction=record)
        rows.append(row)
    head=nn.Sequential(nn.Linear(768,96),nn.GELU(),nn.Linear(96,3))
    head.load_state_dict(torch.load(ROOT/'checkpoints'/f'modernbert_head_{seed}.pt',weights_only=True)); head.eval()
    @torch.inference_mode()
    def infer(r):
        if numeric_case(r): return execute(extract(r['prompt'])[1])
        feed=tok(r['prompt'],return_tensors='pt',truncation=False)
        x=encoder(**feed).last_hidden_state
        pooled=(x*feed.attention_mask[:,:,None]).sum(1)/feed.attention_mask.sum(1,keepdim=True)
        return LABELS[int(head(pooled).argmax(-1))]
    dump(ROOT/'results'/f'modernbert_guarded_{seed}.json',dict(model='modernbert_guarded',seed=seed,
        training='Same frozen-encoder learned head; numeric routing and parser are handwritten',
        metrics=metrics(rows),predictions=rows,latency=latency(infer,data),environment=environment(),
        limitations=['Numeric parser supports the generated input grammar; not a general semantic extractor',
                    'Not a purely learned arithmetic module and not a universal fixed checkpoint']))
    print('SAVED guarded',seed,metrics(rows)['all'],flush=True)
