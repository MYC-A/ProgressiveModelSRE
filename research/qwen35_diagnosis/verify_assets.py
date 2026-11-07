"""Verify pinned local assets without loading the model or large files into RAM."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = Path('A:/ProgressiveModelSRE-cache/models/qwen35_08')
metadata = json.loads((CACHE / 'upstream_metadata.json').read_text(encoding='utf-8'))
checks = {}
names = {'config.json', 'merges.txt', 'tokenizer.json', 'tokenizer_config.json',
         'vocab.json', 'model.safetensors-00001-of-00001.safetensors'}
for asset in metadata['siblings']:
    name = asset['rfilename']
    if name not in names:
        continue
    path = CACHE / name
    if asset.get('lfs'):
        with path.open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        expected = asset['lfs']['sha256']
    else:
        data = path.read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        expected = asset['blobId']
    assert actual == expected, name
    checks[name] = dict(matches=True, actual=actual, expected=expected)
template_path = ROOT / 'upstream_chat_template.jinja'
url = f"https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/{metadata['sha']}/chat_template.jinja"
if not template_path.exists():
    template_path.write_bytes(urllib.request.urlopen(url, timeout=30).read())
data = template_path.read_bytes()
asset = next(a for a in metadata['siblings'] if a['rfilename'] == 'chat_template.jinja')
assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == asset['blobId']
embedded = json.loads((CACHE / 'tokenizer_config.json').read_text(encoding='utf-8'))['chat_template']
assert data.decode() == embedded
checks['upstream_template'] = dict(matches_embedded=True, url=url, sha256=hashlib.sha256(data).hexdigest())
(ROOT / 'asset_verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
print('VERIFIED', len(names), 'local assets and pinned official template')
