"""Download and verify original Loghub BGL, preserving real sequence order."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
import requests

OUT=Path('A:/ProgressiveModelSRE-cache/datasets')
OUT.mkdir(exist_ok=True,parents=True)
metadata=requests.get('https://zenodo.org/api/records/8196385',timeout=30)
metadata.raise_for_status()
spec=next(f for f in metadata.json()['files'] if f['key']=='BGL.zip')
path=OUT/'BGL.zip'
if not path.exists():
    with requests.get(spec['links']['self'],stream=True,timeout=(15,45)) as r:
        r.raise_for_status()
        with path.with_suffix('.partial').open('wb') as f:
            for block in r.iter_content(1048576): f.write(block)
    path.with_suffix('.partial').rename(path)
assert 'md5:'+hashlib.file_digest(path.open('rb'),'md5').hexdigest()==spec['checksum']
(OUT/'BGL.source.json').write_text(json.dumps(spec,indent=2))
with zipfile.ZipFile(path) as z:
    assert z.testzip() is None
    for name in z.namelist():
        if name.endswith('.log') and not (OUT/Path(name).name).exists():
            with z.open(name) as src,(OUT/Path(name).name).open('wb') as dst:
                shutil.copyfileobj(src,dst)
print('BGL_VERIFIED_READY',path)
