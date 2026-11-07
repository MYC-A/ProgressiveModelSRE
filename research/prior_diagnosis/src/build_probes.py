import json, hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PREV=ROOT.parent/'t5_family'

def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2))

def main():
    allrows=json.loads((PREV/'results/lamini_flan_large/hard_predictions.json').read_text())
    seeds=[c for c in allrows if c['view']=='raw' and c['id'] in ['p19_fail','p19_ok','p20_fail','p20_ok','p24_fail','p24_ok']]
    targets=json.loads((PREV/'data/target_hypotheses.json').read_text())
    template=json.loads((PREV/'protocol.json').read_text())['primary_prompt']
    probes=[]
    def add(id,group,prompt,expected,kind='state',**extra):
        probes.append(dict(id=id,group=group,prompt=prompt,expected=expected,kind=kind,**extra))
    def fields(c):
        d=targets[c['pair_id']]
        return dict(target=c['target'],ok=d['ok'],failure=d['failure'],text=c['input_text'])
    def full(c,text=None):
        f=fields(c)
        if text is not None:f['text']=text
        return template.format(**f)
    lookup={c['id']:c for c in seeds}
    for c in seeds:
        f=fields(c);id=c['id'];gold=c['gold'];text=c['input_text']
        original=full(c)
        variants={
            'original':original,
            'without_criteria':'\n'.join(x for x in original.splitlines() if not x.startswith(('Success criterion:','Failure criterion:'))),
            'criteria_after_text':original.split('\nTarget:')[0]+f"\nTarget: {f['target']}\nText: {text}\nSuccess criterion: {f['ok']}\nFailure criterion: {f['failure']}\nAnswer:",
            'failure_criterion_first':original.split('\nTarget:')[0]+f"\nTarget: {f['target']}\nFailure criterion: {f['failure']}\nSuccess criterion: {f['ok']}\nText: {text}\nAnswer:",
            'short_question':text+'\nWhat is the current outcome? Answer with one word: success, failure, or unknown.',
            'evidence_first':full(c,'. '.join(text.rstrip('.').split('. ')[::-1])+'.'),
        }
        if id.startswith('p19'):
            observed='503' if gold=='FAIL' else '200';expected='200'
            compact=f'Expected HTTP status: {expected}. Observed HTTP status: {observed}. If observed equals expected, answer success. Otherwise, answer failure. Answer:'
            actualq='What HTTP status was actually returned? Answer with the number only.'
            expectedq='What HTTP status was expected? Answer with the number only.'
            yesno='Does the observed HTTP status match the expected HTTP status? Answer yes or no.'
        elif id.startswith('p20'):
            observed='2' if gold=='FAIL' else '0';expected='0'
            compact=f'Successful exit code: {expected}. Actual exit code: {observed}. If actual equals successful, answer success. Otherwise, answer failure. Answer:'
            actualq='What exit code did the command actually return? Answer with the number only.'
            expectedq='Which exit code means success? Answer with the number only.'
            yesno='Does the actual exit code indicate success under the stated rule? Answer yes or no.'
        else:
            observed='0' if gold=='FAIL' else '1';expected='1'
            compact=f'Successful status: {expected}. Actual status: {observed}. If actual equals successful, answer success. Otherwise, answer failure. Answer:'
            actualq='What status did the task actually return? Answer with the number only.'
            expectedq='Which status means success in this application? Answer with the number only.'
            yesno='Does the actual status mean success under the stated rule? Answer yes or no.'
        variants['compact_fields']=compact
        variants['given_comparison']=compact.rsplit('Answer:',1)[0]+f'The values are {"equal" if gold=="OK" else "different"}. Answer:'
        for name,prompt in variants.items():add(id+'__'+name,name,prompt,gold,seed=id,observed=observed,expected_value=expected)
        for name,q,answer in [('extract_actual',actualq,observed),('extract_expected',expectedq,expected),('rule_yes_no',yesno,'yes' if gold=='OK' else 'no')]:
            add(id+'__'+name,name,text+'\n'+q,answer,'number' if name.startswith('extract') else 'yesno',seed=id)
        add(id+'__explanation','explanation',text+'\nApply the stated rule to the observed value. Explain the comparison briefly, then state success or failure.',gold,'explanation',seed=id,max_new_tokens=64)
    # Balanced value counterfactuals. Expected and observed values independently vary.
    for family,values,seed in [('exit',[0,1,2],'p20_ok'),('http',[200,404,503],'p19_ok'),('custom',[0,1],'p24_ok')]:
        c=lookup[seed]
        for expected in values:
            for actual in values:
                gold='OK' if actual==expected else 'FAIL'
                if family=='exit':
                    text=f'For this command, exit code {expected} means success and every other exit code means failure. The command exited with code {actual}.'
                elif family=='http':
                    text=f'Production GET /checkout returned HTTP {actual}. The expected response status is HTTP {expected}.'
                else:
                    text=f'In this application, status {expected} means success and status {1-expected} means failure. Task result: status {actual}.'
                for format,prompt in [('full',full(c,text)),('fields',f'Expected value: {expected}. Observed value: {actual}. Matching values mean success. Different values mean failure. Answer success or failure:')]:
                    add(f'sweep_{family}_{expected}_{actual}_{format}','sweep_'+format,prompt,gold,family=family,expected_value=expected,observed=actual)
    # Pure equality, separately from domain vocabulary and from state labels.
    pairs=[(0,2),(2,0),(0,0),(2,2),(200,503),(503,200),(200,200),(503,503),(7,9),(9,7),(7,7),(9,9)]
    for a,b in pairs:
        add(f'equality_{a}_{b}','pure_equality',f'Are the numbers {a} and {b} equal? Answer yes or no.','yes' if a==b else 'no','yesno',a=a,b=b)
        add(f'mapping_{a}_{b}','pure_mapping',f'Expected value: {a}. Observed value: {b}. If the numbers are equal, answer success. If they are different, answer failure. Answer:','OK' if a==b else 'FAIL',a=a,b=b)
    dump(ROOT/'data/seeds.json',seeds)
    dump(ROOT/'data/probes.json',probes)
    registry=json.loads((PREV/'data/selected_models.json').read_text())
    selected=[{**r,'local_path':str(PREV/'models'/r['alias'])}for r in registry if r['alias']in ['lamini_flan_large','flan_large']]
    dump(ROOT/'data/models.json',selected)
    dump(ROOT/'data/previous_originals.json',{r['alias']:[c for c in json.loads((PREV/'results'/r['alias']/'hard_predictions.json').read_text())if c['view']=='raw'and c['id']in lookup]for r in selected})
    protocol={
        'frozen_before_new_outputs':True,'date':'2026-11-07','purpose':'Controlled diagnosis of three previously observed LaMini failures; not an independent accuracy benchmark',
        'n_probes':len(probes),'models':selected,'generation':{'dtype':'float32','threads':4,'batch':8,'greedy_max_new_tokens':24,'explanation_max_new_tokens':64,'input_limit':512},
        'supplementary':{'beam_search':'num_beams=4 on six original prompts only','single_item_replication':'three original failing prompts, greedy','candidate_likelihood':'teacher-forced sum logP of Success./Failure./Unknown. including EOS on originals and compact_fields; normalized over these three strings only, not calibrated confidence'},
        'questions':['Are old wrong outputs reproducible?','Can actual and expected values be extracted separately?','Can equality be answered without domain labels?','Does clause/criterion order or simplification change the result?','Does beam search remove the error?','Does the parent checkpoint behave differently?'],
        'limitations':['Diagnostic probes constructed after observing errors','Prompt changes are interventions on input behavior, not a mechanistic proof about weights or attention','Explanations are generated outputs, not faithful internal traces','No training or selection of an operational threshold','Given-comparison variant supplies part of the answer and is only a diagnostic oracle'],
        'probes_sha256':hashlib.sha256((ROOT/'data/probes.json').read_bytes()).hexdigest()}
    dump(ROOT/'protocol.json',protocol)
    print('FROZEN',len(probes),'probes',protocol['probes_sha256'])
if __name__=='__main__':main()
