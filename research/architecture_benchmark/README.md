# Architecture benchmark pilot

Two separate tasks: criterion-conditioned SRE text classification and supervised
alert classification of real chronological BGL windows. See
[comparison_report_ru.md](comparison_report_ru.md) for measured results and limits.

The test data is frozen. Do not regenerate it to improve observed results.
`protocol.json` records the SRE file hashes. `stream/protocol.json` records the
original BGL checksum, chronological boundaries and window selection.

Models and original BGL live outside this repository under
`A:/ProgressiveModelSRE-cache`. Change that explicit cache path in the scripts
if reproducing on another machine. No credentials or hosted API are required.

Using the workspace virtual environment and existing dependency lock:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r research\requirements.lock.txt requests==2.34.2
.\.venv\Scripts\python.exe research\architecture_benchmark\download_candidates.py
.\.venv\Scripts\python.exe research\architecture_benchmark\prepare_bgl.py
.\.venv\Scripts\python.exe research\architecture_benchmark\verify_ssd.py
.\.venv\Scripts\python.exe research\architecture_benchmark\train_text.py
.\.venv\Scripts\python.exe research\architecture_benchmark\run_comparison.py
.\.venv\Scripts\python.exe research\architecture_benchmark\guarded.py
.\.venv\Scripts\python.exe research\architecture_benchmark\stream_operating_points.py
.\.venv\Scripts\python.exe research\architecture_benchmark\verify_evidence.py
.\.venv\Scripts\python.exe research\architecture_benchmark\build_report.py
.\.venv\Scripts\python.exe research\architecture_benchmark\plot_results.py
.\.venv\Scripts\python.exe research\architecture_benchmark\package_results.py
```

The original FLAN/LaMini controls also require pinned cached weights from the
earlier research (`research/download_models.py`, revisions in
`research/prior_diagnosis/data/models.json`). All candidate download metadata is
pinned to the revisions in the checked-in `*.metadata.json` files. Large weights
must match the upstream SHA256 before becoming available to the benchmark.

`build_data.py` creates synthetic inputs, and `stream_data.py` constructs BGL
features. They were run before the respective candidate outputs. Use the supplied
frozen inputs for reproduction. To regenerate, use a separate clean experiment
directory, keeping the previous outputs and protocol unchanged.

The trained small networks are prototypes, not pretrained public detector
checkpoints. The SSD implementation has input-dependent state parameters and
was checked against an independent recurrence. Its dense CPU implementation
does not measure the optimized production throughput of official Mamba-2.

ModernBERT's encoder is frozen; only its classification head is trained. Numeric
feature and guarded variants use an explicitly limited parser. The parser's
grammar is known and synthetic: these variants do not establish general-purpose
extraction, arithmetic reasoning or production reliability.

The complete original BGL contains 4,747,963 lines. The pilot uses 3000/800/1000
non-overlapping windows from chronological train/validation/test portions.
Original alert labels and severity fields are excluded from the model features.
Repeated message templates can still occur in different time periods.
