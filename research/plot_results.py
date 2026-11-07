"""Standalone figures; vocabulary projections are explicitly labelled diagnostic."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10, 'figure.dpi':150})

fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for alias,label,color in [('lamini_flan_large','LaMini','#c54141'),('flan_large','FLAN-T5','#2467a6')]:
    for axis,case in zip(axes,['200_200','200_503']):
        path=ROOT/'results'/alias/f'trace_{case}.json'
        if not path.exists(): continue
        data=json.loads(path.read_text(encoding='utf-8'))
        axis.plot([r['layer'] for r in data['decoder_logit_lens']],
                  [r['yes_minus_no'] for r in data['decoder_logit_lens']],label=label,color=color)
        axis.set_title(case.replace('_',' = ')+'?')
        axis.set_xlabel('Decoder block (one-based)')
        axis.set_ylabel('log P(Yes/yes) - log P(No/no)')
        axis.axhline(0,color='gray',lw=.8)
        axis.legend()
fig.suptitle('Diagnostic intermediate vocabulary projection (not a causal explanation)')
fig.savefig(OUT/'decoder_projections.png')
fig.savefig(OUT/'decoder_projections.svg')
plt.close(fig)

path=ROOT/'results/lamini_flan_large/ff_ablation_screen.json'
if path.exists():
    data=json.loads(path.read_text(encoding='utf-8'))
    fig,axis=plt.subplots(figsize=(9,4),layout='constrained')
    for case,label,color in [('equality_200_200','200 = 200','#258442'),('equality_200_503','200 = 503','#cf7b21')]:
        axis.plot([r['layers'][0] for r in data],
                  [next(a['yes_minus_no'] for a in r['answers'] if a['id']==case) for r in data],
                  marker='.',label=label,color=color)
    axis.axhline(0,color='gray',lw=.8)
    axis.set(xlabel='Encoder FF update omitted (one-based block)',ylabel='log P(Yes/yes) - log P(No/no)',
             title='Actual intervention: omit one encoder FF residual update')
    axis.legend()
    fig.savefig(OUT/'encoder_ff_ablations.png')
    fig.savefig(OUT/'encoder_ff_ablations.svg')
    plt.close(fig)

paths=[ROOT/'results/lamini_flan_large'/f'{name}.json'
       for name in ['baseline','ff_skip_21','ff_skip_23','ff_skip_21_23']]
if all(p.exists() for p in paths):
    datasets=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
    fig,axis=plt.subplots(figsize=(8,4),layout='constrained')
    for shift,case,label,color in [(-.17,'equality_200_200','200 = 200','#258442'),
                                   (.17,'equality_200_503','200 = 503','#cf7b21')]:
        values=[next(r['yes_minus_no'] for r in rs if r['id']==case) for rs in datasets]
        axis.bar([i+shift for i in range(4)],values,width=.32,label=label,color=color)
    axis.axhline(0,color='black',lw=.8)
    axis.set_xticks(range(4),['Baseline','Omit FF21','Omit FF23','Omit FF21 + FF23'])
    axis.set(ylabel='log P(Yes/yes) - log P(No/no)',title='The joint intervention changes the equal-pair answer')
    axis.legend()
    fig.savefig(OUT/'ff_pair_intervention.png')
    fig.savefig(OUT/'ff_pair_intervention.svg')
    plt.close(fig)
