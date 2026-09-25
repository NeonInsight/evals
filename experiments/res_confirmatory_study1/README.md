# RES Confirmatory Study 1

Matched 2×2 within-fixture behavioral study: 64 canonical fixtures × neutral/explicit framing × low/high relational complexity = 256 Qwen observations. See [preregistration.md](preregistration.md) for frozen endpoints and interpretation limits. The prior five-factor capability gate remains failed.

## Reproduce the freeze and preflight

From the repository root, before any scored inference:

```bash
python -m unittest discover -s experiments/res_confirmatory_study1 -p 'test_*.py' -v
python experiments/res_confirmatory_study1/run.py freeze
python experiments/res_confirmatory_study1/run.py preflight
python -m pip install transformers==4.48.3 tokenizers==0.21.0
python experiments/res_confirmatory_study1/run.py token-audit
```

The freeze command refuses an existing manifest or results directory. The checked-in `manifest.json` hashes every canonical fixture, rendered prompt, and source file. `prompts.jsonl` holds all 256 messages in their frozen randomized order. `results/token_counts.json` records exact pinned-tokenizer counts and chat-template hashes before inference. Store the manifest commit before changing `control.json` to `run`.

## Run and analyze

The GitHub workflow `.github/workflows/res-confirmatory-study1.yml` listens only to `control.json` pushes on `res-confirmatory-study-1`. A `preflight` push validates the frozen data and pinned-tokenizer audit without loading the model. A later `run` push starts 32 independent, resumable shards. No workflow advances itself from preflight to inference. Run attempt increments on a fresh commit for checkpointed recovery; do not use the Actions Re-run button on an earlier snapshot.

Local equivalent (requires pinned CPU PyTorch 2.5.1, Transformers 4.48.3, NumPy 1.26.4, SciPy 1.14.1, pandas 2.2.3, statsmodels 0.14.5, safetensors 0.5.2, scikit-learn 1.6.1):

```bash
python experiments/res_confirmatory_study1/run.py preflight
python experiments/res_confirmatory_study1/run.py token-audit
for shard in $(seq 0 31); do python experiments/res_confirmatory_study1/run.py run --shard "$shard"; done
python experiments/res_confirmatory_study1/analyze.py
```

The workflow uploads individual raw shard checkpoints even if inference fails, collects without overwriting observations, and commits `results/` only. Review `results/run_status.json`, `results/results.json`, and `results/REPORT.md`. A partial study is explicitly marked incomplete and receives no confirmatory analysis. The raw shard JSON is the record of generated text, IDs, logits, prompt provenance, and parameters. `scored_observations.csv`, `matched_transitions.csv`, `factorial_analysis.json`, `decision_margin_analysis.json`, and `length_confound_analysis.json` are derived artifacts. Verify hashes with `python experiments/res_confirmatory_study1/run.py preflight`.
