"""Additional threshold diagnostics, chosen only from validation normal scores."""
import json
import numpy as np
import torch
from common import ROOT,dump
from networks import Classifier
from stream_data import DIM

validation=np.load(ROOT/'stream'/'validation.npz')
x=torch.from_numpy(validation['x']); y=validation['y']
results={}
for path in sorted((ROOT/'results').glob('stream_*.json')):
    data=json.loads(path.read_text())
    model=Classifier(data['model'],d=32,input_dim=DIM,classes=2)
    model.load_state_dict(torch.load(ROOT/'checkpoints'/(path.stem+'.pt'),weights_only=True)); model.eval()
    with torch.inference_mode():
        p=torch.cat([model(b).softmax(-1)[:,1] for b in x.split(64)]).numpy()
    assert abs(float(np.quantile(p[y==0],.99,method='higher'))-data['threshold'])<1e-7
    scores=np.array([r['score'] for r in data['predictions']])
    labels=np.array([r['expected']=='failure' for r in data['predictions']])
    points=[]
    for budget in [.005,.01,.03,.05]:
        threshold=float(np.quantile(p[y==0],1-budget,method='higher'))
        predicts=scores>threshold
        points.append(dict(validation_fpr_budget=budget,threshold=threshold,
            test_false_alarm=int((predicts&~labels).sum()),
            test_missed_failure=int((~predicts&labels).sum()),test_failure_n=int(labels.sum()),test_success_n=int((~labels).sum())))
    # Pairwise AUC definition handles score ties without an external dependency.
    positive=scores[labels][:,None]; negative=scores[~labels][None,:]
    auc=float(((positive>negative)+.5*(positive==negative)).mean())
    results[path.stem]=dict(roc_auc=auc,points=points)
dump(ROOT/'stream_operating_points.json',dict(
    purpose='Additional diagnostic operating points after initial 1% results; no model retraining or test-based threshold selection',
    results=results))
print('OPERATING_POINTS_VERIFIED',len(results))
