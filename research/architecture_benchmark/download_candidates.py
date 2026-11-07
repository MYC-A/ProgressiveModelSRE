"""Pinned, SHA256-verified candidate downloads using the existing ranged downloader."""
import json
from pathlib import Path
import sys
import requests

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import download_models as downloader

specs=[]
for alias,repo in [('modernbert','answerdotai/ModernBERT-base'),('qwen3_06','Qwen/Qwen3-0.6B'),
                   ('qwen35_08','Qwen/Qwen3.5-0.8B')]:
    metadata=json.loads((ROOT/(repo.split('/')[-1]+'.metadata.json')).read_text())
    spec=dict(alias=alias,repo=repo,revision=metadata['sha'])
    if alias=='qwen35_08':
        spec['weight_filename']='model.safetensors-00001-of-00001.safetensors'
    specs.append(spec)

def run(spec):
    # The Qwen3.5 repository contains a single indexed shard. A regular file is
    # created from the exact pinned shard; safetensors keys stay unchanged.
    result=downloader.download(spec)
    if spec['alias']=='qwen35_08':
        out=Path(result['local_path'])
        shard=out/'model.safetensors-00001-of-00001.safetensors'
        target=out/'model.safetensors'
        if not target.exists():
            # Hard link, same exact file bytes, no extra full-size allocation.
            target.hardlink_to(shard)
    out=Path(result['local_path'])
    metadata=json.loads((out/'upstream_metadata.json').read_text())
    names={f['rfilename'] for f in metadata['siblings']}
    for name in ('vocab.json','merges.txt','added_tokens.json'):
        if name in names:
            response=requests.get(f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{name}",timeout=(15,40))
            response.raise_for_status()
            (out/name).write_bytes(response.content)
    return result

results=[]
# Bound total network workers: the shared range downloader uses 16 per model.
for spec in specs:
    results.append(run(spec))
    (ROOT/'models.local.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('ALL_CANDIDATES_VERIFIED',flush=True)
