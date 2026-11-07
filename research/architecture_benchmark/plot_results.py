"""Export figures of measured quality and warm single-item latency."""
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT

groups=defaultdict(list)
for p in (ROOT/'results').glob('*.json'):
    if p.stem.endswith('_diagnostic'): continue
    d=json.loads(p.read_text(encoding='utf-8'))
    groups[('BGL' if p.stem.startswith('stream_') else 'SRE',d['model'])].append(d)
out=ROOT/'figures'; out.mkdir(exist_ok=True)
for task in ['SRE','BGL']:
    selected=sorted([(name,rs) for (track,name),rs in groups.items() if track==task],
                    key=lambda x:np.mean([r['metrics']['all']['accuracy'] for r in x[1]]))
    if not selected: continue
    fig,axes=plt.subplots(1,3,figsize=(14,max(4,len(selected)*.37)),layout='constrained',sharey=True)
    for i,(name,runs) in enumerate(selected):
        ms=[r['metrics']['all'] for r in runs]
        quality=[m['accuracy']*100 if task=='SRE' else 100*(1-m['missed_failure']/m['failure_n']) for m in ms]
        errors=[m['missed_failure'] if task=='SRE' else 100*m['false_alarm']/m['success_n'] for m in ms]
        speed=[r['latency']['p95_ms'] for r in runs]
        for ax,values,color in zip(axes,[quality,errors,speed],['#287a63','#bc5743','#356b9c']):
            average=float(np.mean(values)); low,high=min(values),max(values)
            ax.errorbar(average,i,xerr=[[max(0,average-low)],[max(0,high-average)]],fmt='o',color=color,capsize=3)
            ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(len(selected)),[name for name,_ in selected])
    axes[0].set_xlabel('Accuracy (%)' if task=='SRE' else 'Alert recall (%)'); axes[0].set_xlim(0,103)
    axes[1].set_xlabel('Missed failures / 144' if task=='SRE' else 'False positive rate (%)')
    axes[2].set_xlabel('Warm single-item p95 (ms, log scale)'); axes[2].set_xscale('log')
    fig.suptitle(task+': pilot results; ranges across seeds; CPU shared desktop')
    fig.savefig(out/(task.lower()+'.png'),dpi=160); fig.savefig(out/(task.lower()+'.svg'))
    plt.close(fig)
print('FIGURES_EXPORTED',out)
