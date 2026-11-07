"""Reproduce, trace, and causally intervene on pinned LaMini / FLAN-T5 checkpoints.

Layer numbers in result files are one-based. No training or checkpoint mutation.
Run each model separately so a CPU machine need not retain both models in RAM.
"""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0', HF_HUB_DISABLE_TELEMETRY='1')
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
CACHE = Path('A:/ProgressiveModelSRE-cache/models')
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.manual_seed(0)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def freeze():
    prior = json.loads((ROOT / 'prior_diagnosis/data/probes.json').read_text())
    rows = [{**r, 'suite': r['group']} for r in prior if r['group'] in ('pure_equality', 'original')]
    # Fixed before reading outputs. Original 12 examples are discovery data only.
    rng = random.Random(20261107)
    values = [3, 4, 10, 17, 42, 100, 201, 404, 999, 1000, 12345, 54321]
    for a in values:
        b = rng.choice([v for v in values if v != a])
        for actual in (a, b):
            for template, prompt in {
                'original_form': f'Are the numbers {a} and {actual} equal? Answer yes or no.',
                'same': f'Is {a} the same number as {actual}? Answer yes or no.',
                'statement': f'{a} is equal to {actual}. Is this statement true? Answer yes or no.',
                'inequality': f'Are the numbers {a} and {actual} different? Answer yes or no.',
            }.items():
                equal = a == actual
                expected = 'yes' if (not equal if template == 'inequality' else equal) else 'no'
                rows.append(dict(id=f'new_{template}_{a}_{actual}', suite=f'heldout_{template}',
                                 prompt=prompt, expected=expected, kind='yesno', a=a, b=actual))
    # Names isolate symbol identity from number-specific semantics.
    for a, b in [('apple','apple'), ('apple','pear'), ('pear','pear'), ('pear','apple'),
                 ('alpha','alpha'), ('alpha','beta'), ('beta','beta'), ('beta','alpha')]:
        rows.append(dict(id=f'words_{a}_{b}', suite='word_identity',
                         prompt=f'Are the words {a} and {b} identical? Answer yes or no.',
                         expected='yes' if a==b else 'no', kind='yesno'))
    dump(ROOT / 'probes.frozen.json', rows)
    dump(ROOT / 'protocol.json', dict(
        frozen_before_new_outputs=True, date='2026-11-07', n=len(rows),
        probes_sha256=hashlib.sha256((ROOT/'probes.frozen.json').read_bytes()).hexdigest(),
        model_specs=json.loads((ROOT/'prior_diagnosis/data/models.json').read_text()),
        dtype='float32', threads=4, decoding='greedy', attention='eager',
        primary_metric='exact normalized complete generated answer',
        screening_metric='full-vocabulary argmax at first answer token; not forced binary classification',
        interventions=['skip encoder FF residual update, all 24 individually and layers 21+23 jointly',
                       'replace encoder DenseReluDense weights only with parent checkpoint, and reciprocal',
                       'replace encoder FF layer norms only',
                       'within-model encoder activation patch, matched-length sources 2=2, 2!=7, 7=7 to 200=200 and 200!=7'],
        limitations=['New numbers and templates are a diagnostic holdout, not an independent benchmark',
                     'Ablations and hybrids change the computation; success is not proof of a unique natural circuit',
                     'Attention weights and intermediate vocabulary projections alone do not establish causality']))
    return rows


def label_ids(tok):
    result = {}
    for label in ('yes', 'no'):
        ids = set()
        for variant in (label, label.title()):
            encoded = tok(variant, add_special_tokens=False).input_ids
            assert len(encoded) == 1, (variant, encoded)
            ids.add(encoded[0])
        result[label] = sorted(ids)
    return result


def stats(logits, tok, ids):
    logits = logits.float()
    probs = logits.softmax(-1)
    yes = torch.logsumexp(logits[ids['yes']], -1)
    no = torch.logsumexp(logits[ids['no']], -1)
    vals, top = probs.topk(6)
    return dict(first_token=tok.decode([int(logits.argmax())]).strip(),
                yes_minus_no=float(yes-no),
                p_yes=float(probs[ids['yes']].sum()), p_no=float(probs[ids['no']].sum()),
                top6=[dict(id=int(i), token=tok.convert_ids_to_tokens(int(i)), p=float(p)) for i,p in zip(top,vals)])


def normalize(s, kind):
    s = s.strip().strip('.!:;').strip().lower()
    if kind == 'state':
        return {'success':'OK', 'failure':'FAIL', 'unknown':'UNKNOWN'}.get(s, 'INVALID')
    return s if s in ('yes','no') else 'INVALID'


@torch.inference_mode()
def evaluate(model, tok, ids, rows, generate=False):
    answers = []
    for start in range(0, len(rows), 8):
        batch = rows[start:start+8]
        feed = tok([r['prompt'] for r in batch], padding=True, return_tensors='pt')
        if generate:
            result = model.generate(**feed, max_new_tokens=8, do_sample=False, num_beams=1,
                                    return_dict_in_generate=True, output_scores=True)
            logits = result.scores[0]
            texts = tok.batch_decode(result.sequences, skip_special_tokens=True)
            output_ids = result.sequences.tolist()
        else:
            result = model(**feed, decoder_input_ids=torch.zeros(len(batch),1,dtype=torch.long), use_cache=False)
            logits = result.logits[:,0]
            texts = [tok.decode([int(v.argmax())]).strip() for v in logits]
            output_ids = [[int(v.argmax())] for v in logits]
        for j,(row, v, text, generated_ids) in enumerate(zip(batch, logits, texts, output_ids)):
            prediction = normalize(text, row['kind'])
            answers.append({**row, **stats(v,tok,ids), 'output_text':text,
                            'input_token_ids':feed.input_ids[j][feed.attention_mask[j].bool()].tolist(),
                            'output_token_ids':generated_ids,
                            'complete_generation':generate,
                            'hit_token_limit':bool(generate and len(generated_ids)-1>=8 and tok.eos_token_id not in generated_ids),
                            'prediction':prediction, 'correct':prediction==row['expected']})
    return answers


def summarize(rows):
    by = defaultdict(list)
    for row in rows:
        by[row['suite']].append(row)
    return {key:dict(n=len(rs), correct=sum(r['correct'] for r in rs),
                     predictions=dict(Counter(r['prediction'] for r in rs)),
                     expected_yes=dict(n=sum(r['expected']=='yes' for r in rs),
                                       correct=sum(r['correct'] for r in rs if r['expected']=='yes')),
                     expected_no=dict(n=sum(r['expected']=='no' for r in rs),
                                      correct=sum(r['correct'] for r in rs if r['expected']=='no')))
            for key,rs in by.items()}


@contextmanager
def skip_ff(model, layers):
    handles = [model.encoder.block[i-1].layer[-1].register_forward_hook(lambda m,a,o: a[0]) for i in layers]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def replace_weights(model, donor, group, layers=range(1,25)):
    selected = set(layers)
    saved = {}
    # Open only the exact donor subsets. Mapping two full checkpoints can
    # exceed Windows' commit limit even when their pages are not resident.
    readers = {i: safe_open(str(donor/'ff_donors'/f'{i:02d}.safetensors'),
                            framework='pt', device='cpu') for i in selected}
    try:
        with torch.no_grad():
            for name,p in model.named_parameters():
                if not name.startswith('encoder.block.'):
                    continue
                block = int(name.split('.')[2])+1
                match = '.DenseReluDense.' in name if group=='dense' else '.layer.1.layer_norm.' in name
                if block in selected and match:
                    # Preserve original storage and swap tensor references. This avoids
                    # mutating mapped checkpoint pages or allocating a second FF copy.
                    saved[name] = (p.data, p.data_ptr())
                    source = readers[block].get_tensor(name)
                    assert source.shape == p.shape and source.dtype == p.dtype
                    p.data = source
        assert saved, group
        print('SWAPPED', group, len(saved), 'tensors', flush=True)
        yield
    finally:
        with torch.no_grad():
            params = dict(model.named_parameters())
            for name,(value,pointer) in saved.items():
                params[name].data = value
                assert params[name].data_ptr() == pointer
        del saved, readers
        gc.collect()


@torch.inference_mode()
def trace(model,tok,ids,prompt,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    feed = tok(prompt,return_tensors='pt')
    tokens = tok.convert_ids_to_tokens(feed.input_ids[0])
    saved = {}
    hooks = []
    for stack in ('encoder','decoder'):
        for i,block in enumerate(getattr(model,stack).block):
            def hook(m,a,o,key=f'{stack}.block.{i+1}'):
                value = o[0] if isinstance(o,(tuple,list)) else o
                saved[key] = value.detach().float().clone()
            hooks.append(block.register_forward_hook(hook))
    try:
        result=model(**feed,decoder_input_ids=torch.zeros(1,1,dtype=torch.long),
                     output_hidden_states=True,output_attentions=True,use_cache=False)
    finally:
        for h in hooks: h.remove()
    for stack, attentions in [('encoder_attention',result.encoder_attentions),
                              ('decoder_attention',result.decoder_attentions),
                              ('cross_attention',result.cross_attentions)]:
        for i,v in enumerate(attentions): saved[f'{stack}.{i+1}']=v.detach().float().clone()
    saved['input_ids']=feed.input_ids
    saved['logits']=result.logits.detach().float()
    assert all(torch.isfinite(v).all() for v in saved.values()), 'Nonfinite raw trace'
    torch.save(saved,path.with_suffix('.pt'))
    lens=[]
    for i in range(1,25):
        h=model.decoder.final_layer_norm(saved[f'decoder.block.{i}'])
        if model.config.scale_decoder_outputs: h=h*(model.model_dim**-0.5)
        v=model.lm_head(h)[0,0]
        lens.append(dict(layer=i,**stats(v,tok,ids)))
    final_error=float((model.lm_head(model.decoder.final_layer_norm(saved['decoder.block.24']))
                       *(model.model_dim**-0.5 if model.config.scale_decoder_outputs else 1)-result.logits).abs().max())
    assert final_error < 1e-4, final_error
    numeric_positions=[i for i,s in enumerate(tokens) if s.lstrip('▁').isdigit()]
    numeric_links=[]
    if len(numeric_positions)==2:
        a,b=numeric_positions
        for layer,attention in enumerate(result.encoder_attentions,1):
            for head in range(attention.shape[1]):
                forward=float(attention[0,head,a,b])
                backward=float(attention[0,head,b,a])
                numeric_links.append(dict(layer=layer,head=head+1,first_receives_second=forward,
                                          second_receives_first=backward,bidirectional_sum=forward+backward))
        numeric_links.sort(key=lambda r:r['bidirectional_sum'],reverse=True)
    dump(path.with_suffix('.json'),dict(prompt=prompt,tokens=tokens,input_ids=feed.input_ids[0].tolist(),
                                      final=stats(result.logits[0,0],tok,ids),decoder_logit_lens=lens,
                                      encoder_numeric_attention_top=numeric_links[:8],
                                      final_projection_max_abs_error=final_error,all_saved_tensors_finite=True))


@torch.inference_mode()
def activation_patches(model,tok,ids,out):
    # 503 and 0 split into multiple tokens. Use aligned one-token arguments
    # for activation patch controls, avoiding a positional/length confound.
    targets=['Are the numbers 200 and 200 equal? Answer yes or no.',
             'Are the numbers 200 and 7 equal? Answer yes or no.']
    rows=[]
    for a,b in [(2,2),(2,7),(7,7)]:
        source=f'Are the numbers {a} and {b} equal? Answer yes or no.'
        cache={}
        hooks=[]
        for i,block in enumerate(model.encoder.block):
            def hook(m,a,o,key=i):
                cache[key]=(o[0] if isinstance(o,(tuple,list)) else o).detach().clone()
            hooks.append(block.register_forward_hook(hook))
        source_feed=tok(source,return_tensors='pt')
        try: source_result=model(**source_feed,decoder_input_ids=torch.zeros(1,1,dtype=torch.long),use_cache=False)
        finally:
            for h in hooks: h.remove()
        source_margin=stats(source_result.logits[0,0],tok,ids)['yes_minus_no']
        # Identify numeric positions from the actual tokenizer, rather than hard-coded indices.
        source_numbers=[i for i,s in enumerate(tok.convert_ids_to_tokens(source_feed.input_ids[0])) if s.lstrip('▁').isdigit()]
        assert len(source_numbers)==2
        for target in targets:
            feed=tok(target,return_tensors='pt')
            positions=[i for i,s in enumerate(tok.convert_ids_to_tokens(feed.input_ids[0])) if s.lstrip('▁').isdigit()]
            assert len(positions)==2 and feed.input_ids.shape==source_feed.input_ids.shape
            for i in range(24):
                for scope in ('numbers','all'):
                    def patch(m,a,o,i=i,scope=scope):
                        h=o[0] if isinstance(o,(tuple,list)) else o
                        v=h.clone()
                        if scope=='all': v[:]=cache[i]
                        else: v[:,positions]=cache[i][:,source_numbers]
                        if isinstance(o,tuple): return (v,)+o[1:]
                        if isinstance(o,list): return [v]+o[1:]
                        return v
                    handle=model.encoder.block[i].register_forward_hook(patch)
                    try:
                        result=model(**feed,decoder_input_ids=torch.zeros(1,1,dtype=torch.long),use_cache=False)
                        entry=dict(target=target,source=source,layer=i+1,scope=scope,**stats(result.logits[0,0],tok,ids))
                        if i==23 and scope=='all':
                            assert abs(entry['yes_minus_no']-source_margin)<1e-4, 'Whole encoder patch did not reproduce source'
                        rows.append(entry)
                    finally: handle.remove()
            dump(out/'activation_patches.json',rows)
        print('ACTIVATION_PATCH_SOURCE_DONE',a,b,flush=True)
    return rows


def neuron_attribution(model,tok,ids,out,rows):
    """Local gradient screening, followed by actual neuron ablation and controls."""
    model.requires_grad_(False)
    captures={}
    handles=[]
    for layer in (21,23):
        def capture(m,a,layer=layer):
            a[0].requires_grad_(True)
            captures[layer]=a[0]
        handles.append(model.encoder.block[layer-1].layer[-1].DenseReluDense.wo.register_forward_pre_hook(capture))
    feed=tok('Are the numbers 200 and 200 equal? Answer yes or no.',return_tensors='pt')
    try:
        result=model(**feed,decoder_input_ids=torch.zeros(1,1,dtype=torch.long),use_cache=False)
        v=result.logits[0,0].float()
        objective=torch.logsumexp(v[ids['yes']],-1)-torch.logsumexp(v[ids['no']],-1)
        grads=torch.autograd.grad(objective,[captures[l] for l in (21,23)])
        records=[]
        for layer,grad in zip((21,23),grads):
            contribution=(grad*captures[layer]).detach()[0]
            score=contribution.sum(0)
            values,indices=score.sort()
            records.append(dict(layer=layer,local_grad_times_activation_sum=float(score.sum()),
                                most_negative=[dict(neuron=int(n),score=float(s)) for n,s in zip(indices[:32],values[:32])],
                                token_contributions=contribution.sum(-1).tolist()))
        dump(out/'neuron_gradient_screen.json',dict(
            prompt=tok.decode(feed.input_ids[0]),tokens=tok.convert_ids_to_tokens(feed.input_ids[0]),
            layers=records,interpretation='Local derivatives, not proof of individual neuron causality; neuron indices zero-based'))
    finally:
        for h in handles: h.remove()
        del result, captures, grads
        gc.collect()
    for count in (1,8,32):
        handles=[]
        for record in records:
            positions=[r['neuron'] for r in record['most_negative'][:count]]
            def zero(m,a,positions=positions):
                value=a[0].clone()
                value[:,:,positions]=0
                return (value,)+a[1:]
            handles.append(model.encoder.block[record['layer']-1].layer[-1].DenseReluDense.wo.register_forward_pre_hook(zero))
        try: answers=evaluate(model,tok,ids,rows,generate=True)
        finally:
            for h in handles: h.remove()
        dump(out/f'neuron_skip_{count}_per_layer.json',answers)
        print('NEURON_ABLATION',count,json.dumps(summarize(answers)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['lamini_flan_large','flan_large'])
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--stage',choices=['baseline','mechanism','heads','positions','controls','hybrid'],default='baseline')
    args=parser.parse_args()
    if args.freeze:
        freeze()
        return
    rows=json.loads((ROOT/'probes.frozen.json').read_text())
    out=OUT/args.model
    out.mkdir(parents=True,exist_ok=True)
    path=CACHE/args.model
    started=time.monotonic()
    tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
    model=AutoModelForSeq2SeqLM.from_pretrained(str(path),local_files_only=True,dtype=torch.float32,
                                             attn_implementation='eager',weights_only=True).eval()
    ids=label_ids(tok)
    dump(out/'runtime.json',dict(torch=torch.__version__,transformers=__import__('transformers').__version__,
                                config=model.config.to_dict(),label_ids=ids,device=str(model.device),
                                shared_encoder_tied=model.shared.weight.data_ptr()==model.encoder.embed_tokens.weight.data_ptr(),
                                shared_decoder_tied=model.shared.weight.data_ptr()==model.decoder.embed_tokens.weight.data_ptr(),
                                shared_lmhead_tied=model.shared.weight.data_ptr()==model.lm_head.weight.data_ptr()))
    print('LOADED',args.model,'stage',args.stage,'seconds',round(time.monotonic()-started,1),flush=True)
    if args.stage=='baseline':
        answers=evaluate(model,tok,ids,rows,generate=True)
        dump(out/'baseline.json',answers)
        dump(out/'baseline_summary.json',summarize(answers))
        for a,b in [(200,200),(200,503),(2,2),(2,0)]:
            trace(model,tok,ids,f'Are the numbers {a} and {b} equal? Answer yes or no.',out/f'trace_{a}_{b}')
        print('BASELINE',json.dumps(summarize(answers)),flush=True)
    elif args.stage=='mechanism':
        discovery=[r for r in rows if r['suite']=='pure_equality']
        screening=[]
        for layers in ([i] for i in range(1,25)):
            with skip_ff(model,layers): answers=evaluate(model,tok,ids,discovery)
            screening.append(dict(layers=layers,summary=summarize(answers),answers=answers))
            dump(out/'ff_ablation_screen.json',screening)
            print('ABLATE',layers,summarize(answers),flush=True)
        for layers in ([21,23],[22,24],[20,22],[21],[23]):
            with skip_ff(model,layers): answers=evaluate(model,tok,ids,rows,generate=True)
            dump(out/('ff_skip_'+'_'.join(map(str,layers))+'.json'),answers)
            print('VALIDATED_SKIP',layers,json.dumps(summarize(answers)),flush=True)
        activation_patches(model,tok,ids,out)
        neuron_attribution(model,tok,ids,out,rows)
    elif args.stage=='controls':
        with skip_ff(model,[21,23]):
            answers=evaluate(model,tok,ids,rows,generate=True)
        dump(out/'ff_skip_21_23.json',answers)
        print('PARENT_PAIR_CONTROL',json.dumps(summarize(answers)),flush=True)
    elif args.stage=='positions':
        # Adaptive follow-up to the measured gradients: localize FF effects to
        # numerical input pieces vs the rest of the prompt. Same frozen probes.
        numeric_lookup=torch.tensor([(tok.convert_ids_to_tokens(i) or '').lstrip('▁').isdigit()
                                     for i in range(model.config.vocab_size)],dtype=torch.bool)
        context={}
        def capture_input(m,a,kw):
            input_ids=kw.get('input_ids',a[0] if a else None)
            context['numeric']=numeric_lookup[input_ids]
        outer=model.encoder.register_forward_pre_hook(capture_input,with_kwargs=True)
        actions=[[(21,'numbers')],[(21,'other')],[(21,'numbers'),(23,'all')],
                 [(21,'all'),(23,'numbers')],[(21,'all'),(23,'other')]]
        try:
            for selection in actions:
                handles=[]
                for layer,scope in selection:
                    def selective_skip(m,a,o,scope=scope):
                        if scope=='all': return a[0]
                        mask=context['numeric'] if scope=='numbers' else ~context['numeric']
                        return torch.where(mask.unsqueeze(-1),a[0],o)
                    handles.append(model.encoder.block[layer-1].layer[-1].register_forward_hook(selective_skip))
                try: answers=evaluate(model,tok,ids,rows,generate=True)
                finally:
                    for h in handles: h.remove()
                name='position_skip_'+'_'.join(f'{l}_{s}' for l,s in selection)
                dump(out/(name+'.json'),answers)
                print('POSITION_ABLATION',selection,json.dumps(summarize(answers)),flush=True)
        finally: outer.remove()
    elif args.stage=='heads':
        trace_data=json.loads((out/'trace_200_200.json').read_text(encoding='utf-8'))
        selected=[(r['layer'],r['head']) for r in trace_data['encoder_numeric_attention_top'][:2]]
        for heads in [[selected[0]],[selected[1]],selected]:
            handles=[]
            for layer,head in heads:
                def zero_head(m,a,head=head):
                    value=a[0].clone()
                    width=model.config.d_kv
                    value[:,:,(head-1)*width:head*width]=0
                    return (value,)+a[1:]
                handles.append(model.encoder.block[layer-1].layer[0].SelfAttention.o.register_forward_pre_hook(zero_head))
            try: answers=evaluate(model,tok,ids,rows,generate=True)
            finally:
                for h in handles: h.remove()
            name='head_skip_'+'_'.join(f'{l}_{h}' for l,h in heads)
            dump(out/(name+'.json'),answers)
            print('HEAD_ABLATION',heads,json.dumps(summarize(answers)),flush=True)
    else:
        donor=CACHE/('flan_large' if args.model=='lamini_flan_large' else 'lamini_flan_large')
        for group,layers in [('dense',list(range(1,25))),('norm',list(range(1,25))),('dense',[21,23])]:
            with replace_weights(model,donor,group,layers):
                answers=evaluate(model,tok,ids,rows,generate=True)
            name='swap_'+group+('_21_23' if len(layers)==2 else '_all')
            dump(out/(name+'.json'),answers)
            print(name,json.dumps(summarize(answers)),flush=True)
        # Restoration control includes actual and counterexample inputs.
        restored=evaluate(model,tok,ids,[r for r in rows if r['suite']=='pure_equality'],generate=True)
        dump(out/'restoration_control.json',restored)
        base=json.loads((out/'baseline.json').read_text())
        original={r['id']:r for r in base}
        assert all(r['output_text']==original[r['id']]['output_text'] and
                   abs(r['yes_minus_no']-original[r['id']]['yes_minus_no'])<1e-4 for r in restored)
    print('DONE',args.model,args.stage,'seconds',round(time.monotonic()-started,1),flush=True)


if __name__=='__main__':
    main()
