import json,re,hashlib
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def semantic(row):
    # Supplementary answer-format interpretation; strict results remain intact.
    # Applied identically to both checkpoints, with evidence retained verbatim.
    if row['kind']=='number':
        nums=set(re.findall(r'(?<!\w)\d+(?!\w)',row['output_text']))
        return next(iter(nums))if len(nums)==1 else 'INVALID'
    if row['kind']=='state':
        s=row['output_text'].strip().lower()
        m=re.fullmatch(r'(?:the\s+)?(?:current\s+)?(?:outcome|result|status)(?:\s+of\s+(?:the\s+)?(?:target|command|task|request))?(?:\s+is|\s*:)\s+(success|failure|unknown)[.!]?',s)
        if m:return {'success':'OK','failure':'FAIL','unknown':'UNKNOWN'}[m.group(1)]
    return row['prediction']
def main():
    summary={};allrows={};verification={}
    probes=json.loads((ROOT/'data/probes.json').read_text());ids=[p['id']for p in probes]
    assert len(ids)==len(set(ids))
    for spec in json.loads((ROOT/'data/models.json').read_text()):
        alias=spec['alias'];folder=ROOT/'results'/alias
        rows=json.loads((folder/'predictions.json').read_text())
        assert [r['id']for r in rows]==ids
        counts=defaultdict(lambda:Counter())
        for r in rows:
            r['semantic_prediction']=semantic(r)
            r['semantic_correct']=r['semantic_prediction']==r['expected']if r['kind']!='explanation'else None
            if r['kind']=='explanation':continue
            c=counts[r['group']];c['n']+=1;c['strict_correct']+=r['correct'];c['semantic_correct']+=r['semantic_correct'];c['strict_invalid']+=r['prediction']=='INVALID'
            c['FAIL_expected']+=r['expected']=='FAIL';c['OK_expected']+=r['expected']=='OK'
        beam=json.loads((folder/'beam4.json').read_text())
        singles=json.loads((folder/'single_replication.json').read_text())
        prev=json.loads((ROOT/'data/previous_originals.json').read_text())[alias]
        prior={c['id']:c for c in prev}
        originals=[r for r in rows if r['group']=='original']
        verification[alias]={'all_probe_ids_present':True,'original_reproduction':[{'id':r['seed'],'same_output':r['output_text']==prior[r['seed']]['output_text']}for r in originals],
            'single_reproduction':singles,'beam4':beam,'runtime':json.loads((folder/'runtime.json').read_text())}
        summary[alias]={k:dict(v)for k,v in counts.items()};allrows[alias]=rows
        byid={r['id']:r for r in rows};chained=[]
        for original in originals:
            seed=original['seed'];actual=byid[seed+'__extract_actual']['semantic_prediction'];expected=byid[seed+'__extract_expected']['semantic_prediction']
            state=('OK'if actual==expected else 'FAIL')if actual!='INVALID'and expected!='INVALID'else 'UNKNOWN'
            chained.append({'seed':seed,'actual':actual,'expected':expected,'operator':'eq (provided by the diagnostic task definition)','prediction':state,'gold':original['expected'],'correct':state==original['expected']})
        dump(folder/'extraction_then_comparison.json',chained)
        dump(folder/'predictions_evaluated.json',rows)
    verification['probes_hash_matches']=hashlib.sha256((ROOT/'data/probes.json').read_bytes()).hexdigest()==json.loads((ROOT/'protocol.json').read_text())['probes_sha256']
    verification['weights']=json.loads((ROOT/'results/weight_verification.json').read_text())
    fpath=ROOT/'results/followup_predictions.json'
    if fpath.exists():
        follow=json.loads(fpath.read_text());base=json.loads((ROOT/'results/lamini_flan_large/predictions.json').read_text())
        equality={r['id']:r for r in base if r['group']=='pure_equality'}
        verification['followup']={'input_hash_matches':hashlib.sha256((ROOT/'data/followup_probes.json').read_bytes()).hexdigest()==json.loads((ROOT/'followup_protocol.json').read_text())['inputs_sha256'],
            'equality_single_vs_batch':[{'id':r['id'],'same_output':r['output_text']==equality[r['id']]['output_text']}for r in follow if r['group']=='single_equality_replication'],
            'same_followup_ids':{r['id']for r in follow}=={r['id']for r in json.loads((ROOT/'data/followup_probes.json').read_text())}}
    dump(ROOT/'results/summary.json',summary);dump(ROOT/'results/verification.json',verification)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
