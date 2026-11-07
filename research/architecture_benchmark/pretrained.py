"""Frozen ModernBERT heads and real greedy generation from pinned local candidates."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0')
import argparse
import copy
import gc
import re
import time
from decimal import Decimal
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModel,AutoModelForCausalLM,AutoModelForSeq2SeqLM,Qwen3_5ForConditionalGeneration
from common import ROOT,CACHE,LABELS,load,dump,metrics,latency,environment,verify_inputs
from numeric import extract,execute

def modernbert():
    path=CACHE/'modernbert'; start=time.monotonic()
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModel.from_pretrained(path,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
    for p in model.parameters(): p.requires_grad_(False)
    data={s:load(s) for s in ['train','validation','test']}
    features={}; numeric={}
    @torch.inference_mode()
    def feature(rows):
        feed=tok([r['prompt'] for r in rows],padding=True,return_tensors='pt',truncation=False)
        h=model(**feed).last_hidden_state
        return (h*feed.attention_mask[:,:,None]).sum(1)/feed.attention_mask.sum(1,keepdim=True)
    for split,rows in data.items():
        cache=ROOT/'checkpoints'/f'modernbert_features_{split}.pt'; cache.parent.mkdir(exist_ok=True)
        if cache.exists(): features[split]=torch.load(cache,weights_only=True)
        else:
            blocks=[]
            for at in range(0,len(rows),8):
                blocks.append(feature(rows[at:at+8]).clone())
                if at%160==0: print('FEATURES',split,at,'/',len(rows),flush=True)
            features[split]=torch.cat(blocks).clone(); torch.save(features[split],cache)
        numeric[split]=torch.tensor(np.stack([extract(r['prompt'])[0] for r in rows]))
    labels={s:torch.tensor([LABELS.index(r['expected']) for r in rs]) for s,rs in data.items()}
    for modular in [False,True]:
        name='modernbert_numeric' if modular else 'modernbert_head'
        for seed in [0,1,2]:
            torch.manual_seed(seed)
            x={s:torch.cat([v,numeric[s]],-1) if modular else v for s,v in features.items()}
            head=nn.Sequential(nn.Linear(x['train'].shape[-1],96),nn.GELU(),nn.Linear(96,3))
            optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
            best=-1; history=[]; stale=0
            for epoch in range(60):
                head.train()
                for ids in torch.randperm(len(x['train'])).split(128):
                    optimizer.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(head(x['train'][ids]),labels['train'][ids])
                    loss.backward(); optimizer.step()
                head.eval()
                with torch.no_grad(): score=float((head(x['validation']).argmax(-1)==labels['validation']).float().mean())
                history.append(dict(epoch=epoch+1,validation_accuracy=score))
                if score>best:
                    best=score; state=copy.deepcopy(head.state_dict()); stale=0
                else: stale+=1
                if stale>=8: break
            head.load_state_dict(state); head.eval()
            with torch.inference_mode():
                ps=head(x['test']).argmax(-1).tolist()
                rows=[dict(**r,prediction=LABELS[p]) for r,p in zip(data['test'],ps)]
                def infer(row):
                    v=feature([row])
                    if modular: v=torch.cat([v,torch.tensor(extract(row['prompt'])[0])[None]],-1)
                    return LABELS[int(head(v).argmax(-1))]
                timing=latency(infer,data['test'])
            torch.save(state,ROOT/'checkpoints'/f'{name}_{seed}.pt')
            dump(ROOT/'results'/f'{name}_{seed}.json',dict(model=name,seed=seed,
                training='Frozen pretrained ModernBERT; only MLP classification head trained on synthetic data',
                modular_features='schema-limited parser and exact numeric relation features' if modular else None,
                history=history,metrics=metrics(rows),latency=timing,predictions=rows,environment=environment(),
                seconds_including_shared_feature_extraction=time.monotonic()-start))
            print('SAVED',name,seed,metrics(rows)['all'],flush=True)
    rows=[]
    for r in data['test']:
        _,record=extract(r['prompt']); rows.append(dict(**r,prediction=execute(record),extraction=record))
    dump(ROOT/'results'/'numeric_executor.json',dict(model='numeric_executor',
        training='None; deliberately limited handwritten schema parser',metrics=metrics(rows),predictions=rows,
        latency=latency(lambda r:execute(extract(r['prompt'])[1]),data['test']),environment=environment()))

def normalize(text,kind):
    text=text.strip().strip('.!:;').strip().lower()
    allowed=['yes','no'] if kind=='yesno' else LABELS
    return text if text in allowed else 'INVALID'

def generative(alias):
    start=time.monotonic(); path=CACHE/alias
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    causal=alias.startswith('qwen')
    if causal: tok.padding_side='left'
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    cls=(Qwen3_5ForConditionalGeneration if alias=='qwen35_08' else
         AutoModelForCausalLM if causal else AutoModelForSeq2SeqLM)
    model=cls.from_pretrained(path,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
    print('LOADED',alias,round(time.monotonic()-start,2),flush=True)
    def prompt(r):
        if not causal: return r['prompt']
        return tok.apply_chat_template([dict(role='user',content=r['prompt'])],tokenize=False,
                                       add_generation_prompt=True,enable_thinking=False)
    @torch.inference_mode()
    def infer(batch,rich=False):
        feed=tok([prompt(r) for r in batch],padding=True,return_tensors='pt',truncation=False)
        output=model.generate(**feed,max_new_tokens=8,do_sample=False,num_beams=1,
                              pad_token_id=tok.pad_token_id,use_cache=True)
        generated=output[:,feed.input_ids.shape[1]:] if causal else output[:,1:]
        texts=tok.batch_decode(generated,skip_special_tokens=True)
        if not rich: return [normalize(t,r['kind']) for t,r in zip(texts,batch)]
        return [dict(**r,prediction=normalize(t,r['kind']),output_text=t,output_ids=ids.tolist(),
                     hit_token_limit=bool(len(ids)>=8 and tok.eos_token_id not in ids.tolist()),
                     input_token_ids=feed.input_ids[j][feed.attention_mask[j].bool()].tolist())
                for j,(r,t,ids) in enumerate(zip(batch,texts,generated))]
    for split in ['test','diagnostic']:
        data=load(split); rows=[]
        batch_size=4 if alias=='flan_large' else 8
        for at in range(0,len(data),batch_size):
            rows+=infer(data[at:at+batch_size],True)
            if at%80==0: print('GENERATING',alias,split,at,'/',len(data),flush=True)
        timing=latency(lambda r:infer([r]),data,n=24)
        dump(ROOT/'results'/f'{alias}_{split}.json',dict(model=alias,training='No task-specific training',
             metrics=metrics(rows),latency=timing,predictions=rows,environment=environment(),
             seconds=time.monotonic()-start,batch_size=batch_size,
             decoding='greedy, max 8 new tokens; non-thinking chat template for Qwen'))
        print('SAVED',alias,split,metrics(rows)['all'],flush=True)
    if alias=='qwen35_08':
        baseline=load('results/qwen35_08_test')['predictions']
        def supported_numeric(r):
            match=re.search(r'^Success criterion: (.*)$',r['prompt'],re.M)
            return match and re.search(r'code|latency|throughput',match[1].lower())
        rows=copy.deepcopy(baseline)
        for r in rows:
            if supported_numeric(r):
                _,record=extract(r['prompt'])
                for key in ['output_text','output_ids','input_token_ids','hit_token_limit']:
                    if key in r: r['baseline_'+key]=r.pop(key)
                r.update(prediction=execute(record),extraction=record,method='schema numeric executor')
            else: r['method']='Qwen3.5 direct classification'
            r['system_output']=r['prediction']
        def guarded_infer(r):
            if supported_numeric(r): return execute(extract(r['prompt'])[1])
            return infer([r])[0]
        dump(ROOT/'results'/'qwen35_guarded.json',dict(model='qwen35_guarded',
             training='No task-specific neural training; handwritten schema numeric executor with Qwen3.5 fallback',
             metrics=metrics(rows),predictions=rows,latency=latency(guarded_infer,load('test'),n=24),
             environment=environment(),limitations=['Parser knows generated numeric input grammar',
             'This is a hybrid system, not proof of general text extraction or a corrected neural checkpoint']))
        print('SAVED qwen35_guarded',metrics(rows)['all'],flush=True)
    if alias=='flan_large':
        # Actual neural extraction + exact execution, not oracle numeric fields.
        # Non-numeric cases retain the same baseline model decision.
        baseline=load('results/flan_large_test')['predictions']
        selected=[]
        for i,r in enumerate(baseline):
            fields=dict(re.findall(r'^(Target|Success criterion|Text): (.*)$',r['prompt'],re.M))
            c=fields['Success criterion']
            if re.search('code|latency|throughput',c.lower()): selected.append((i,fields))
        @torch.inference_mode()
        def extract_batch(items):
            prompts=[]
            for _,f in items:
                prompts.append(f"Text: {f['Text']}\nWhat numeric value was actually observed for {f['Target']} in the current run? Answer only the number, or unknown if it is not established.")
                prompts.append(f"Success criterion: {f['Success criterion']}\nWhat is the numeric reference value in this criterion? Answer only the number.")
            feed=tok(prompts,padding=True,return_tensors='pt',truncation=False)
            outputs=model.generate(**feed,max_new_tokens=8,do_sample=False,num_beams=1)
            return tok.batch_decode(outputs,skip_special_tokens=True)
        def resolve(actual,expected,criterion):
            texts=[t.strip().strip('.').strip() for t in (actual,expected)]
            if not all(re.fullmatch(r'-?\d+(?:\.\d+)?',t) for t in texts): return 'unknown'
            a,e=map(Decimal,texts); c=criterion.lower()
            if any(t in c for t in ['at most','no greater','not exceeding']): ok=a<=e
            elif any(t in c for t in ['at least','no smaller','reaching or exceeding']): ok=a>=e
            else: ok=a==e
            return 'success' if ok else 'failure'
        rows=copy.deepcopy(baseline)
        for at in range(0,len(selected),4):
            batch=selected[at:at+4]; values=extract_batch(batch)
            for j,(index,fields) in enumerate(batch):
                actual,expected=values[2*j:2*j+2]
                for key in ['output_text','output_ids','input_token_ids','hit_token_limit']:
                    if key in rows[index]: rows[index]['baseline_'+key]=rows[index].pop(key)
                rows[index].update(prediction=resolve(actual,expected,fields['Success criterion']),
                    extraction=dict(actual_output=actual,expected_output=expected),
                    method='FLAN two numeric extraction queries + schema-limited exact operator')
                rows[index]['system_output']=rows[index]['prediction']
            if at%64==0: print('EXTRACTION',at,'/',len(selected),flush=True)
        def hybrid_infer(r):
            fields=dict(re.findall(r'^(Target|Success criterion|Text): (.*)$',r['prompt'],re.M))
            if re.search('code|latency|throughput',fields['Success criterion'].lower()):
                values=extract_batch([(0,fields)])
                return resolve(*values,fields['Success criterion'])
            return infer([r])[0]
        dump(ROOT/'results'/'flan_extract_execute.json',dict(model='flan_extract_execute',
             training='No task-specific training; two actual extraction calls for numeric cases',
             metrics=metrics(rows),predictions=rows,latency=latency(hybrid_infer,load('test'),n=24),
             environment=environment(),limitations=['Operator recognition is schema-specific',
             'Extracted values are not independently verified for target/time binding','No oracle actual/expected values used']))
        print('SAVED flan_extract_execute',metrics(rows)['all'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('model'); args=parser.parse_args()
    verify_inputs()
    if args.model=='modernbert': modernbert()
    else: generative(args.model)
