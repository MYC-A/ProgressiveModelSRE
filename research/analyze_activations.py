import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(2)
summary={}
for alias in ['lamini_flan_large','flan_large']:
    for path in (ROOT/'results'/alias).glob('trace_*.pt'):
        tensors=torch.load(path,weights_only=True,map_location='cpu')
        records=[]
        for stack in ['encoder','decoder']:
            for layer in range(1,25):
                h=tensors[f'{stack}.block.{layer}'].float()
                record=dict(stack=stack,layer=layer,rms=float(h.square().mean().sqrt()),
                            max_abs=float(h.abs().max()),finite=bool(torch.isfinite(h).all()))
                if stack=='encoder' and path.stem=='trace_200_200':
                    record['numeric_position_cosine']=float(torch.nn.functional.cosine_similarity(h[:,3],h[:,5]).item())
                records.append(record)
        summary[f'{alias}/{path.stem}']=records
(ROOT/'activation_statistics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
for name,records in summary.items():
    if name.endswith('200_200'):
        print(name,[(r['layer'],round(r['rms'],2),round(r['max_abs'],1),round(r.get('numeric_position_cosine',0),5))
                    for r in records if r['stack']=='encoder'])
