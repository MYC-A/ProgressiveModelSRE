import os
for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','DO_NOT_TRACK']:os.environ[key]='1'
os.environ['USE_TF']='0'
import argparse, json, time, hashlib
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

ROOT=Path(__file__).resolve().parents[1]
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def norm(s):return s.strip().strip('.!:;').strip().lower()
def parse(s,kind):
    s=norm(s)
    if kind=='state':return {'success':'OK','failure':'FAIL','unknown':'UNKNOWN'}.get(s,'INVALID')
    if kind=='yesno':return s if s in ['yes','no']else 'INVALID'
    if kind=='number':return s if s.isdigit()else 'INVALID'
    return None
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);args=ap.parse_args()
    spec=next(x for x in json.loads((ROOT/'data/models.json').read_text()) if x['alias']==args.model)
    out=ROOT/'results'/args.model;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    tok=AutoTokenizer.from_pretrained(spec['local_path'],local_files_only=True)
    configs={s['alias']:AutoConfig.from_pretrained(s['local_path'],local_files_only=True).to_dict()for s in json.loads((ROOT/'data/models.json').read_text())}
    dump(out/'effective_configs.json',configs)
    model=AutoModelForSeq2SeqLM.from_pretrained(spec['local_path'],local_files_only=True,dtype=torch.float32,weights_only=True).eval()
    probes=json.loads((ROOT/'data/probes.json').read_text());scores={};start=time.perf_counter()
    print('LOADED',args.model,'probes',len(probes),flush=True)
    jobs={p['prompt']:p.get('max_new_tokens',24)for p in probes}
    for limit in [24,64]:
        prompts=[t for t,v in jobs.items()if v==limit]
        for i in range(0,len(prompts),8):
            batch=prompts[i:i+8];feed=tok(batch,padding=True,truncation=True,max_length=512,return_tensors='pt')
            with torch.inference_mode():
                gen=model.generate(**feed,max_new_tokens=limit,do_sample=False,num_beams=1,return_dict_in_generate=True,output_scores=True)
            first=gen.scores[0].float().softmax(-1)
            for j,(prompt,raw,ids)in enumerate(zip(batch,tok.batch_decode(gen.sequences,skip_special_tokens=True),gen.sequences.tolist())):
                vals,idx=first[j].topk(8)
                scores[prompt]={'output_text':raw,'output_ids':ids,'first_token_top8':[{'id':int(k),'token':tok.convert_ids_to_tokens(int(k)),'p':float(v)}for k,v in zip(idx,vals)],'hit_limit':len(ids)-1>=limit and tok.eos_token_id not in ids}
            print('PROGRESS',args.model,limit,i+len(batch),'/',len(prompts),round(time.perf_counter()-start,1),'sec',flush=True)
            dump(out/'progress.json',{'scores':scores,'elapsed_seconds':time.perf_counter()-start})
    rows=[]
    for p in probes:
        s=scores[p['prompt']];prediction=parse(s['output_text'],p['kind'])
        rows.append({**p,**s,'prediction':prediction,'correct':prediction==p['expected']if prediction is not None else None})
    dump(out/'predictions.json',rows)
    originals=[p for p in probes if p['group']=='original']
    feed=tok([p['prompt']for p in originals],padding=True,return_tensors='pt')
    with torch.inference_mode():beam=model.generate(**feed,max_new_tokens=24,do_sample=False,num_beams=4)
    dump(out/'beam4.json',[{'id':p['id'],'expected':p['expected'],'output_text':s,'prediction':parse(s,'state')}for p,s in zip(originals,tok.batch_decode(beam,skip_special_tokens=True))])
    singles=[]
    for p in originals:
        if p['expected']!='FAIL':continue
        with torch.inference_mode():g=model.generate(**tok(p['prompt'],return_tensors='pt'),max_new_tokens=24,do_sample=False,num_beams=1)
        raw=tok.decode(g[0],skip_special_tokens=True)
        singles.append({'id':p['id'],'output_text':raw,'same_as_batch':raw==scores[p['prompt']]['output_text']})
    dump(out/'single_replication.json',singles)
    candidates=['Success.','Failure.','Unknown.'];label_ids=tok(candidates,padding=True,return_tensors='pt').input_ids
    masked=label_ids.clone();masked[masked==tok.pad_token_id]=-100
    likelihood=[]
    for p in probes:
        if p['group']not in ['original','compact_fields']:continue
        feed=tok([p['prompt']]*3,padding=True,return_tensors='pt')
        with torch.inference_mode():logits=model(**feed,labels=masked).logits.float().log_softmax(-1)
        lp=logits.gather(-1,label_ids.unsqueeze(-1)).squeeze(-1)
        lp=(lp*(masked!=-100)).sum(-1);relative=lp.softmax(-1)
        likelihood.append({'id':p['id'],'candidate_logp':dict(zip(candidates,map(float,lp))),'relative_mass_among_three_strings':dict(zip(candidates,map(float,relative)))})
    dump(out/'candidate_likelihoods.json',likelihood)
    lengths=[len(tok(p['prompt'])['input_ids'])for p in probes]
    dump(out/'runtime.json',{'elapsed_seconds':time.perf_counter()-start,'parameters':sum(p.numel()for p in model.parameters()),'torch':torch.__version__,'dtype':'FP32','threads':4,'max_input_tokens':max(lengths),'truncated_inputs':sum(n>512 for n in lengths),'revision':spec['revision'],'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),'generated_limit_hits':sum(r['hit_limit']for r in rows),'equality_tokenization':{s:tok.convert_ids_to_tokens(tok(s)['input_ids'])for s in ['0','1','2','200','503']}})
    print('DONE',args.model,round(time.perf_counter()-start,1),'seconds',flush=True)
if __name__=='__main__':main()
