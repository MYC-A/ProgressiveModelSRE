"""Bundle code, frozen inputs, trained prototypes and raw results, excluding large upstream weights."""
import hashlib
import json
from pathlib import Path
import zipfile
from common import ROOT

workspace=ROOT.parents[1]
archive=workspace/'architecture_comparison_20261107.zip'
files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST.json']
files += [ROOT.parent/'download_models.py',ROOT.parent/'requirements.lock.txt',ROOT.parent/'probes.frozen.json',
          ROOT.parent/'prior_diagnosis'/'data'/'models.json']
manifest={str(p.relative_to(workspace)).replace('\\','/'):
          dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in files}
path=ROOT/'MANIFEST.json'; path.write_text(json.dumps(manifest,indent=2),encoding='utf-8'); files.append(path)
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files: z.write(p,str(p.relative_to(workspace)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,spec in manifest.items():
        raw=z.read(name)
        assert len(raw)==spec['bytes'] and hashlib.sha256(raw).hexdigest()==spec['sha256'],name
print('VERIFIED_ARCHIVE',archive,archive.stat().st_size,'bytes',len(files),'files')
