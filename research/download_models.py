"""Download the exact checkpoints used in the shared experiment, without credentials."""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parent
CACHE = Path('A:/ProgressiveModelSRE-cache/models')
CHUNK = 32 * 1024 * 1024


def download(spec):
    alias, repo, rev = (spec[k] for k in ('alias', 'repo', 'revision'))
    out = CACHE / alias
    out.mkdir(parents=True, exist_ok=True)
    meta = requests.get(f'https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true', timeout=30)
    meta.raise_for_status()
    files = {f['rfilename']: f for f in meta.json()['siblings']}
    (out / 'upstream_metadata.json').write_text(json.dumps(meta.json(), indent=2), encoding='utf-8')
    names = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'spiece.model']
    names += [spec.get('weight_filename', 'model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin')]
    for name in names:
        if name not in files:
            continue
        target = out / name
        info = files[name]
        expected = info.get('lfs', {}).get('sha256')
        size = info['size']
        if target.exists() and target.stat().st_size == size:
            digest = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
            if not expected or digest == expected:
                print(alias, name, 'already verified', flush=True)
                continue
        url = f'https://huggingface.co/{repo}/resolve/{rev}/{name}'
        if size < CHUNK:
            response = requests.get(url, timeout=(20, 90))
            response.raise_for_status()
            target.write_bytes(response.content)
        else:
            temporary = target.with_suffix(target.suffix + '.partial')
            resuming = temporary.exists()
            if not temporary.exists():
                with temporary.open('wb') as f:
                    f.truncate(size)
            completed_file = target.with_suffix(target.suffix + '.parts.json')
            completed = set(json.loads(completed_file.read_text())) if completed_file.exists() else set()
            parts = list(range((size + CHUNK - 1) // CHUNK))
            start = time.monotonic()

            def part(i):
                part_lo, hi = i * CHUNK, min((i + 1) * CHUNK, size) - 1
                # Previous interrupted requests wrote a valid sequential prefix.
                # Resume conservatively with at least 1 MiB overlap. The final
                # full-file upstream SHA256 remains mandatory before model use.
                lo = part_lo
                if resuming:
                    with temporary.open('rb') as f:
                        for at in range((hi-part_lo)//1048576, -1, -1):
                            f.seek(part_lo+at*1048576)
                            block = f.read(min(1048576,hi-(part_lo+at*1048576)+1))
                            if any(block):
                                lo = part_lo+max(0,at-1)*1048576
                                break
                with requests.Session() as session:
                    while lo <= hi:
                        # Keep each HTTP response short; long responses were being
                        # cut mid-stream on the current network path.
                        end = min(lo+4*1048576-1,hi)
                        for attempt in range(2):
                            n = 0
                            try:
                                with session.get(url + f'?part={i}&start={lo}&end={end}', headers={'Range': f'bytes={lo}-{end}'}, stream=True, timeout=(20, 45)) as r:
                                    r.raise_for_status()
                                    if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {lo}-{end}/{size}':
                                        raise RuntimeError(f'Unexpected range response: {r.status_code} {r.headers.get("Content-Range")}')
                                    with temporary.open('r+b', buffering=0) as f:
                                        f.seek(lo)
                                        for block in r.iter_content(256*1024):
                                            if n + len(block) > end - lo + 1:
                                                raise RuntimeError('Range exceeded expected length')
                                            f.write(block)
                                            n += len(block)
                                    if n != end - lo + 1:
                                        raise RuntimeError('Incomplete range')
                                lo = end+1
                                break
                            except Exception as exc:
                                lo += n
                                print(alias, 'PART_RETRY', i, 'attempt', attempt+1, type(exc).__name__, str(exc)[:160], flush=True)
                                if attempt:
                                    raise
                return i

            with cf.ThreadPoolExecutor(max_workers=16) as pool:
                pending = {pool.submit(part, i) for i in parts if i not in completed}
                failures=[]
                for future in cf.as_completed(pending):
                    try:
                        completed.add(future.result())
                    except Exception as exc:
                        failures.append(exc)
                        print(alias, 'PART_FAILED',type(exc).__name__,str(exc)[:160],flush=True)
                        continue
                    completed_file.write_text(json.dumps(sorted(completed)))
                    print(alias, len(completed), '/', len(parts), 'parts', round(time.monotonic()-start, 1), 'seconds', flush=True)
                if failures:
                    raise RuntimeError(f'{len(failures)} ranges failed; completed ranges retained') from failures[0]
            os.replace(temporary, target)
        digest = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
        if target.stat().st_size != size or (expected and digest != expected):
            raise RuntimeError(f'Integrity verification failed: {target}')
        print(alias, name, size, digest, 'VERIFIED', flush=True)
    return {**spec, 'local_path': str(out)}


if __name__ == '__main__':
    specs = json.loads((ROOT / 'prior_diagnosis/data/models.json').read_text())
    # One model at a time bounds network and disk pressure.
    specs.sort(key=lambda s: s['alias'] != 'lamini_flan_large')
    results = []
    for spec in specs:
        results.append(download(spec))
        (ROOT / 'models.local.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
