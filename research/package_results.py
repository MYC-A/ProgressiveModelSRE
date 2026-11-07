"""Archive research code, raw evidence, and pinned model metadata; omit model weights."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parent
archive=WORKSPACE/'research_results_20261107.zip'
files=[]
for p in ROOT.rglob('*'):
    if (not p.is_file() or '__pycache__' in p.parts or 'training_data_search' in p.parts
            or any(part.startswith('instrumentation') for part in p.parts)):
        continue
    if p.name=='MANIFEST.current.json':
        continue
    if p.parent==ROOT and p.suffix=='.log' and p.name not in ('run.log','heads-lamini.log','positions-lamini.log','download-small-ranges.log','donor-extraction.log','hybrid-lamini.log','hybrid-flan.log','controls-flan.log'):
        continue
    files.append((p,str(p.relative_to(WORKSPACE)).replace('\\','/')))
files.append((WORKSPACE/'README.md','README.md'))
for alias in ('lamini_flan_large','flan_large'):
    p=Path('A:/ProgressiveModelSRE-cache/models')/alias/'upstream_metadata.json'
    files.append((p,f'model_metadata/{alias}.json'))
manifest={name:dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest())
          for p,name in files}
manifest_path=ROOT/'MANIFEST.current.json'
manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
files.append((manifest_path,'research/MANIFEST.current.json'))
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p,name in files:
        z.write(p,name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,spec in manifest.items():
        content=z.read(name)
        assert len(content)==spec['bytes'] and hashlib.sha256(content).hexdigest()==spec['sha256'],name
print('VERIFIED_ARCHIVE',archive,archive.stat().st_size,'bytes',len(files),'files')
