"""Extract exact per-layer FF donor tensors before loading a recipient model.

Windows cannot always reserve two full private checkpoint mappings at once.
Each output contains only three FF matrices and the FF layer norm (~35 MB).
"""
import gc
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

CACHE=Path('A:/ProgressiveModelSRE-cache/models')
torch.set_num_threads(2)
for alias in ('lamini_flan_large','flan_large'):
    path=CACHE/alias
    out=path/'ff_donors'
    out.mkdir(exist_ok=True)
    original=None
    reader=None
    if (path/'model.safetensors').exists():
        reader=safe_open(str(path/'model.safetensors'),framework='pt',device='cpu')
        keys=list(reader.keys())
    else:
        original=torch.load(path/'pytorch_model.bin',mmap=True,weights_only=True,map_location='cpu')
        keys=list(original)
    for layer in range(24):
        prefix=f'encoder.block.{layer}.layer.1.'
        names=[n for n in keys if n.startswith(prefix) and ('.DenseReluDense.' in n or '.layer_norm.' in n)]
        assert len(names)==4,(alias,layer,names)
        selected={n:reader.get_tensor(n) if reader is not None else original[n] for n in names}
        target=out/f'{layer+1:02d}.safetensors'
        save_file(selected,str(target),metadata={'source_model':alias,'layer_one_based':str(layer+1)})
        with safe_open(str(target),framework='pt',device='cpu') as check:
            assert all(torch.equal(check.get_tensor(n),v) for n,v in selected.items())
        del selected
        print('VERIFIED_EXACT_DONOR',alias,layer+1,flush=True)
    del original,reader
    gc.collect()
print('ALL_DONORS_VERIFIED',flush=True)
