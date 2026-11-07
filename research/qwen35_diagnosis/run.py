"""Post-hoc Qwen3.5 diagnosis. Preserve the earlier benchmark and its inputs.

Run with the workspace Python. New prompts are chosen on validation only.
This is an exploratory diagnostic, not an independent production benchmark.
"""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0')
import hashlib
import json
import re
import sys
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / 'architecture_benchmark'
sys.path.insert(0, str(BENCH))
from common import CACHE, metrics, environment, verify_inputs

torch.set_num_threads(4)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(name, value):
    (HERE / (name + '.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fields(row):
    return row['prompt'].split('\n', 1)[1].removesuffix('\nAnswer:')

DEFINITION = ('Classify evidence about the specified target and its CURRENT run into exactly three classes. '
    'success: current evidence for that target proves the success criterion holds. '
    'failure: current evidence for that target proves the success criterion is violated. '
    'unknown: the current outcome is not reported, the evidence is historical, concerns another target, '
    'or reports conflict without resolution. Missing evidence is NOT a failure. '
    'Return exactly one lowercase word: success, failure, or unknown.')
GATE = ('Decide ONLY whether the supplied evidence establishes an unambiguous CURRENT outcome for the specified target. '
    'Both an established successful outcome and an established failed outcome count as yes. '
    'If the current outcome is not reported, the evidence is only historical, concerns a different target, '
    'or contains unresolved conflicting reports, answer no. Reply exactly yes or no.\n')

def canonical(text, mapping=None, thinking=False):
    if thinking:
        if '</think>' not in text:
            return 'NO_FINAL'
        text = text.split('</think>', 1)[1]
    word = text.strip().strip('.!:;').strip().lower()
    return (mapping or {}).get(word, word) if word in (mapping or {'success': 0, 'failure': 0, 'unknown': 0, 'yes': 0, 'no': 0, 'true': 0, 'false': 0}) else 'INVALID'

def select(rows, per_class=2):
    # Deterministic selection independent of predictions, covers missing/other/history/conflict.
    selected = []
    for family in sorted({r['family'] for r in rows}):
        for label in ['success', 'failure', 'unknown']:
            group = sorted([r for r in rows if r['family'] == family and r['expected'] == label], key=lambda r: r['id'])
            if label == 'unknown':
                chosen = [next(r for r in group if int(r['id'].split('_')[-2]) % 4 == k) for k in range(4)]
            else:
                chosen = group[:per_class]
            selected.extend(chosen)
    return selected

def main():
    start = time.monotonic()
    one_phase = '--one-phase' in sys.argv
    verify_inputs()
    validation = select(read(BENCH / 'validation.json'))
    train = read(BENCH / 'train.json')
    examples = [next(r for r in train if r['expected'] == label) for label in ['success', 'failure', 'unknown']]
    frozen = read(BENCH / 'results/qwen35_08_test.json')['predictions']
    panel = select(read(BENCH / 'test.json'))
    wrong = [r for r in frozen if r['expected'] != 'unknown' and r['prediction'] != r['expected']]
    by_id = {r['id']: r for r in panel + wrong}
    panel = list(by_id.values())
    variants = ['baseline', 'system_role', 'definitions', 'label_abc', 'fewshot', 'label_order', 'evidence_gate']
    protocol = dict(post_hoc=True, model='Qwen/Qwen3.5-0.8B', revision=read(BENCH / 'Qwen3.5-0.8B.metadata.json')['sha'],
        original_inputs={str(p.relative_to(BENCH)): digest(p) for p in [BENCH / 'train.json', BENCH / 'validation.json', BENCH / 'test.json', BENCH / 'diagnostic.json']},
        validation_ids=[r['id'] for r in validation], panel_ids=[r['id'] for r in panel],
        examples=examples, variants=variants, definition=DEFINITION, gate=GATE,
        selection='Pick the two non-baseline variants with highest validation accuracy, ties use listed variant order; run once on all frozen 432 test rows.',
        controls=['baseline batch 1 versus batch 8', '8 versus 64 output tokens', 'cache enabled versus disabled',
                  'next-token restricted label probabilities', 'thinking mode with bounded 256 tokens', 'counterfactual equality wording and number size'],
        environment=environment(), limitations=['Prompts and hypotheses chosen after seeing baseline test results',
            'Validation is reused from earlier research; not an independent dataset',
            'Thinking budget is bounded, truncation is a technical non-answer rather than proof of semantic inability'])
    if not (HERE / 'protocol.json').exists():
        save('protocol', protocol)
    else:
        previous = read(HERE / 'protocol.json')
        assert previous == protocol, 'Frozen diagnostic protocol changed'
    path = CACHE / 'qwen35_08'
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tok.padding_side = 'left'
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval()
    print('LOADED', round(time.monotonic() - start, 2), flush=True)

    def messages(row, variant):
        if variant == 'baseline':
            return [dict(role='user', content=row['prompt'])]
        if variant == 'system_role':
            instruction, user = row['prompt'].split('\n', 1)
            return [dict(role='system', content=instruction), dict(role='user', content=user)]
        if variant == 'definitions':
            return [dict(role='user', content=DEFINITION + '\n' + fields(row) + '\nAnswer:')]
        if variant == 'label_abc':
            instruction = DEFINITION.replace('success:', 'A:').replace('failure:', 'B:').replace('unknown:', 'C:')
            instruction = instruction.replace('Return exactly one lowercase word: success, failure, or unknown.', 'Return exactly one capital letter: A, B, or C.')
            return [dict(role='user', content=instruction + '\n' + fields(row) + '\nAnswer:')]
        if variant == 'fewshot':
            out = [dict(role='system', content=DEFINITION)]
            for e in examples:
                out += [dict(role='user', content=fields(e)), dict(role='assistant', content=e['expected'])]
            return out + [dict(role='user', content=fields(row))]
        if variant == 'label_order':
            return [dict(role='user', content=row['prompt'].replace('Answer exactly one word: success, failure, or unknown.', 'Answer exactly one word: unknown, success, or failure.'))]
        if variant == 'evidence_gate':
            return [dict(role='user', content=GATE + fields(row) + '\nAnswer:')]
        raise ValueError(variant)

    @torch.inference_mode()
    def infer(rows, variant='baseline', batch_size=2, max_tokens=8, cache=True, thinking=False, sampling_seed=None):
        result = []
        for at in range(0, len(rows), batch_size):
            batch = rows[at:at + batch_size]
            prompts = [tok.apply_chat_template(messages(r, variant), tokenize=False, add_generation_prompt=True, enable_thinking=thinking) for r in batch]
            feed = tok(prompts, padding=True, return_tensors='pt', truncation=False)
            kwargs = dict(max_new_tokens=max_tokens, do_sample=sampling_seed is not None, num_beams=1, use_cache=cache,
                          pad_token_id=tok.pad_token_id, return_dict_in_generate=True, output_scores=True)
            if sampling_seed is not None:
                torch.manual_seed(sampling_seed + at)
                kwargs.update(temperature=1.0, top_p=1.0, top_k=20)
            t0 = time.monotonic()
            output = model.generate(**feed, **kwargs)
            seconds = time.monotonic() - t0
            ids = output.sequences[:, feed.input_ids.shape[1]:]
            texts = tok.batch_decode(ids, skip_special_tokens=True)
            first = output.scores[0].float().log_softmax(-1)
            labels = ['success', 'failure', 'unknown'] if variant not in ['label_abc', 'evidence_gate'] else ['A', 'B', 'C'] if variant == 'label_abc' else ['yes', 'no']
            label_ids = [tok.encode(s, add_special_tokens=False)[0] for s in labels]
            assert all(len(tok.encode(s, add_special_tokens=False)) == 1 for s in labels)
            mapping = {'a': 'success', 'b': 'failure', 'c': 'unknown'} if variant == 'label_abc' else None
            for j, (r, text, tokens) in enumerate(zip(batch, texts, ids)):
                tokens = tokens.tolist()
                topv, topi = first[j].topk(8)
                result.append(dict(**{k: v for k, v in r.items() if k in ['id', 'suite', 'family', 'expected', 'kind', 'prompt']},
                    prediction=canonical(text, mapping, thinking), output_text=text, output_ids=tokens,
                    hit_token_limit=(tok.eos_token_id not in tokens and len(tokens) >= max_tokens),
                    input_tokens=int(feed.attention_mask[j].sum()), first_label_logprob=dict(zip(labels, first[j, label_ids].tolist())),
                    first_label_conditional=dict(zip(labels, first[j, label_ids].softmax(-1).tolist())),
                    first_top_tokens=[dict(id=int(i), text=tok.decode([int(i)]), logprob=float(v)) for v, i in zip(topv, topi)],
                    batch_seconds=seconds))
            if at % 80 == 0:
                print('INFER', variant, len(rows), at, 'budget', max_tokens, 'cache', cache, 'think', thinking, flush=True)
        return result

    def record(name, rows, **extra):
        save(name, dict(metrics=metrics(rows), predictions=rows, **extra))
        print('SAVED', name, metrics(rows)['all'], flush=True)
        if one_phase:
            sys.exit(0)

    # Baseline panel records probabilities and independently checks archived outputs.
    reference = {r['id']: r for r in frozen}
    if not (HERE / 'panel_baseline.json').exists():
        baseline = infer(panel, batch_size=8)
        record('panel_baseline', baseline, matches_archive=sum(r['output_text'] == reference[r['id']]['output_text'] for r in baseline))
    # Runtime controls use a balanced deterministic slice, plus every observed numeric error.
    control = [next(r for r in panel if r['expected'] == label and r['family'] == family)
               for family in ['binding', 'latency', 'recovery'] for label in ['success', 'failure', 'unknown']]
    control = list({r['id']: r for r in control + wrong}.values())
    for name, kwargs in [('batch1', dict(batch_size=1)), ('long64', dict(max_tokens=64)), ('no_cache', dict(cache=False))]:
        if not (HERE / ('control_' + name + '.json')).exists():
            record('control_' + name, infer(control, **kwargs))

    val_results = {}
    for variant in variants:
        if (HERE / ('validation_' + variant + '.json')).exists():
            val_results[variant] = read(HERE / ('validation_' + variant + '.json'))['predictions']
            continue
        rows = infer(validation, variant)
        if variant == 'evidence_gate':
            baseline_val = {r['id']: r['prediction'] for r in val_results['baseline']}
            for r in rows:
                r['gate_prediction'] = r['prediction']
                r['prediction'] = 'unknown' if r['gate_prediction'] == 'no' else baseline_val[r['id']] if r['gate_prediction'] == 'yes' else 'INVALID'
        val_results[variant] = rows
        record('validation_' + variant, rows)
    order = sorted(variants[1:], key=lambda v: (-metrics(val_results[v])['all']['accuracy'], variants.index(v)))
    chosen = order[:2]
    save('selection', dict(variants=chosen, ranking=[dict(variant=v, accuracy=metrics(val_results[v])['all']['accuracy']) for v in order], method=protocol['selection']))
    for variant in chosen:
        if (HERE / ('test_' + variant + '.json')).exists():
            continue
        rows = infer(read(BENCH / 'test.json'), variant)
        if variant == 'evidence_gate':
            for r in rows:
                r['gate_prediction'] = r['prediction']
                r['prediction'] = 'unknown' if r['gate_prediction'] == 'no' else reference[r['id']]['prediction'] if r['gate_prediction'] == 'yes' else 'INVALID'
        record('test_' + variant, rows, selected_on_validation=True)

    # Equality questions have explicit matched/mismatched controls, several lengths and formats.
    equality = []
    for a, b in [(2, 2), (2, 3), (20, 20), (20, 21), (200, 200), (200, 201), (503, 503), (503, 200), (2000, 2000), (2000, 2001)]:
        for style, prompt, yes, no in [
            ('original', f'Are the numbers {a} and {b} equal? Answer yes or no.', 'yes', 'no'),
            ('simple', f'Is {a} equal to {b}? Reply with only yes or no.', 'yes', 'no'),
            ('symbol', f'Is this statement true or false: {a} == {b}? Reply only true or false.', 'true', 'false')]:
            equality.append(dict(id=f'equality_{style}_{a}_{b}', suite='equality_' + style,
                family='equality', kind='yesno', expected=yes if a == b else no, prompt=prompt))
    save('equality_inputs', equality)
    if not (HERE / 'equality8.json').exists():
        record('equality8', infer(equality))
    subset = [r for r in equality if r['id'] in ['equality_original_200_200', 'equality_original_200_201', 'equality_simple_200_200', 'equality_symbol_200_200']]
    if not (HERE / 'equality64.json').exists():
        record('equality64', infer(subset, max_tokens=64))
    # The source recipe uses sampling. Three fixed seeds measure variance, not just a lucky draw.
    for seed in [0, 1, 2]:
        if not (HERE / ('sampled_' + str(seed) + '.json')).exists():
            record('sampled_' + str(seed), infer(control + subset, max_tokens=64, sampling_seed=seed))
    think_rows = [next(r for r in panel if r['expected'] == 'unknown' and r['family'] == f) for f in ['binding', 'completion', 'recovery']]
    think_rows += wrong[:2] + subset[:1]
    if not (HERE / 'thinking256.json').exists():
        record('thinking256', infer(think_rows, max_tokens=256, thinking=True, batch_size=1))
    verify_inputs()
    save('completion', dict(complete=True, seconds=time.monotonic() - start, environment=environment(),
                            original_inputs_unchanged=True, result_files=sorted(p.name for p in HERE.glob('*.json'))))
    print('COMPLETE', round(time.monotonic() - start, 2), flush=True)

if __name__ == '__main__':
    main()
