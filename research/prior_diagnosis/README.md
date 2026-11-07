# Controlled diagnosis of LaMini-Flan-T5-783M log classification errors

This is a diagnostic follow-up to `t5_family_report_ru.md`, not a new independent benchmark. It investigates three previously observed failures and their matched successful controls.

The frozen 140 prompts include original task prompts, clause/criterion reorderings, shorter questions, compact fields, individual value extraction, yes/no rule checks, numeric counterfactual sweeps, pure equality, and six short explanation requests. Both the LaMini checkpoint and its Google FLAN-T5-large parent are evaluated. No model is trained.

Model revisions and paths are in `data/models.json`. Download the public repositories at those exact revisions into the referenced paths (or change only `local_path` before running). Model weights are excluded from this archive. The archived original seed records include every original prompt, gold state, and previous answer.

Run with Python 3.12 and the dependency versions in `requirements.txt`:

```bash
python src/run_probes.py --model lamini_flan_large
python src/run_probes.py --model flan_large
python src/followup.py
python src/evaluate.py
python src/write_report.py
```

`build_probes.py` recreates inputs from the prior `t5_family` directory. For reproducing inference from this archive, use the already frozen `data/probes.json`; the prior directory is not necessary. Evaluation compares original answers with the archived `data/previous_originals.json`.

`results/*/predictions.json` preserves strict parsing. `predictions_evaluated.json` also includes supplementary explicit-sentence and unique-number interpretation. This supplementary interpretation was added after early outputs showed correct numbers embedded in sentences. It is not substituted for strict format compliance. Explanation outputs are not automatically scored and are not internal reasoning traces.

`candidate_likelihoods.json` contains teacher-forced log likelihoods of three complete answer strings, including EOS. Relative masses are normalized over these three strings only and are not calibrated probabilities of the real-world classes.

The given-comparison variant explicitly supplies equality/difference and is a diagnostic oracle, not an operational solution. Prompt improvements on these selected cases are not claims about the previous full 120-case accuracy or production FPR.
