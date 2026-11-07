import os
for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','DO_NOT_TRACK']:os.environ[key]='1'
os.environ['USE_TF']='0'
import json,hashlib,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def freeze():
    src=json.loads((ROOT/'data/probes.json').read_text());rows=[]
    questions={'exit':'Does the actual exit code indicate success under the stated rule? Answer yes or no.',
      'http':'Does the observed HTTP status match the expected HTTP status? Answer yes or no.',
      'custom':'Does the actual status mean success under the stated rule? Answer yes or no.'}
    for r in src:
        if r['group']=='sweep_full':
            text=r['prompt'].split('\nText: ',1)[1].rsplit('\nAnswer:',1)[0]
            rows.append({'id':r['id']+'__yesno','group':'counterfactual_rule_yesno','prompt':text+'\n'+questions[r['family']],'expected':'yes'if r['expected']=='OK'else 'no','family':r['family']})
        if r['group']=='pure_equality':rows.append({**r,'group':'single_equality_replication'})
    dump(ROOT/'data/followup_probes.json',rows)
    dump(ROOT/'followup_protocol.json',{'frozen_before_followup_outputs':True,'reason':'After main LaMini outputs: check whether the 6/6 domain yes/no result transfers to value counterfactuals; repeat pure equality one prompt per call to exclude batch dependence',
      'model':'lamini_flan_large','n':len(rows),'single_equality':True,'counterfactual_rule_batch':8,'dtype':'FP32','max_new_tokens':24,'do_sample':False,
      'inputs_sha256':hashlib.sha256((ROOT/'data/followup_probes.json').read_bytes()).hexdigest(),'limitation':'Adaptive diagnostic follow-up, not an independent benchmark or a selected production prompt'})
    print('FROZEN followup',len(rows))
def run():
    import torch
    from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    spec=next(s for s in json.loads((ROOT/'data/models.json').read_text())if s['alias']=='lamini_flan_large')
    tok=AutoTokenizer.from_pretrained(spec['local_path'],local_files_only=True)
    model=AutoModelForSeq2SeqLM.from_pretrained(spec['local_path'],local_files_only=True,dtype=torch.float32,weights_only=True).eval()
    rows=json.loads((ROOT/'data/followup_probes.json').read_text());answers={};start=time.perf_counter()
    for group,size in [('counterfactual_rule_yesno',8),('single_equality_replication',1)]:
        qs=[r for r in rows if r['group']==group]
        for a in range(0,len(qs),size):
            batch=qs[a:a+size];feed=tok([r['prompt']for r in batch],padding=True,return_tensors='pt')
            with torch.inference_mode():output=model.generate(**feed,max_new_tokens=24,do_sample=False,num_beams=1)
            for r,s in zip(batch,tok.batch_decode(output,skip_special_tokens=True)):
                norm=s.strip().strip('.!:;').strip().lower();prediction=norm if norm in ['yes','no']else 'INVALID'
                answers[r['id']]={**r,'output_text':s,'prediction':prediction,'correct':prediction==r['expected']}
            print('PROGRESS',group,a+len(batch),'/',len(qs),round(time.perf_counter()-start,1),flush=True)
    out=ROOT/'results/followup_predictions.json';dump(out,[answers[r['id']]for r in rows])
    from collections import Counter
    summary={}
    for group in ['counterfactual_rule_yesno','single_equality_replication']:
        rs=[r for r in answers.values()if r['group']==group]
        summary[group]={'n':len(rs),'correct':sum(r['correct']for r in rs),'outputs':dict(Counter(r['prediction']for r in rs))}
    summary['runtime']={'seconds':time.perf_counter()-start,'scale_decoder_outputs':model.config.scale_decoder_outputs,'tie_word_embeddings':model.config.tie_word_embeddings,'shared_and_lm_head_storage_tied':model.shared.weight.data_ptr()==model.lm_head.weight.data_ptr()}
    dump(ROOT/'results/followup_summary.json',summary);print('DONE',summary,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');args=ap.parse_args()
    freeze()if args.freeze else run()
