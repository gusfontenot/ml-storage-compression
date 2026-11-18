# Experiments

## Record

- History: `experiments/runs.csv` — one row per policy.
- Saved: settings, workload split, data/model hashes, savings, latency, misses, overhead.
- Output: new directory per run; existing logged models stay unchanged.
- Budget: 1,000 µs provisional. All 128 local live decisions exceeded it. Agree on the target before final testing.
- Budget misses: writes for replay; full decisions for live runs.

## 01 — Pilot

- Model: `models/pilot/model.json`.
- Data: 288 VM measurements; three generated workloads.
- Training: random + repetitive. Test: mixed.
- Result: 0% savings; 25% write-budget misses (offline replay).
- Finding: mixed-data compressed size was overpredicted.
- Status: baseline saved; original results imported into the history.

## 02 — More data

- Local pilot check: passed at both quotas; 128 verified writes.
- Live summary: `experiments/local_live_01.json`.
- Raw checks: `artifacts/local_live_01/`.
- Coverage: nine mixes, from repetitive to random.
- Repeats: five. Chunks: eight per workload, 1 MiB each.
- CPU: 25% and 50%, 100 ms period.
- Expected: 2,160 measured codec rows.
- Validation: `mix_3072`. Final test: `mix_1536`.
- Files: `development.csv`, `test.csv`, raw trials, `collection.json`.
- Status: complete on local Ubuntu.
- Saved: `data/local_ubuntu_expanded_01/`.
- Raw trials: `artifacts/local_ubuntu_expanded_01/`.
- Environment and source commit: `data/local_ubuntu_expanded_01/environment.json`.

Next collection: same machine, new output directory:

```bash
.venv/bin/python benchmarks/collect_pilot.py --expanded --output artifacts/local_ubuntu_expanded_02
```

Require `collection.json` status `complete`. Timing spikes stay in the data.

## 03 — Validation

```bash
.venv/bin/python -m compression_ml.selector data/local_ubuntu_expanded_01/development.csv \
  --feature-set pilot --holdout-workload mix_3072 --role validation \
  --budget-us 1000 --output models/local_v2_validation
```

- Status: complete; 560 training decisions, 80 validation decisions.
- Training: seven mixes. Validation: one entire mix, all quotas and repeats.
- Result: ML chose Zstandard; 24.9% savings; 100% missed the 1 ms write budget.
- LZ4 was faster in this validation.
- Size error, LZ4: 256,542 → 131,032 bytes; Zstandard: 260,818 → 133,962 bytes.
- Comparison: original VM-trained model vs new locally trained model, same local validation data.
- Compare: size/time prediction errors, savings, budget misses, fixed codecs.
- Tune if needed: `--max-depth`, `--min-samples-leaf`; new output per attempt.
- Final test: keep `test.csv` unused during tuning.

## 04 — Final test

Pending: agree on the budget and choose settings using validation. Example command:

```bash
.venv/bin/python -m compression_ml.selector data/local_ubuntu_expanded_01/development.csv \
  --feature-set pilot --test-data data/local_ubuntu_expanded_01/test.csv \
  --budget-us 1000 --output models/v2_final
```

- Use the chosen tree settings; defaults are depth 5, leaf size 2.
- Training: all development workloads. Test: reserved workload.
- One final evaluation; further tuning needs a new reserved test set.
- These are held-out generated mix levels, not independent real datasets.

## 05 — Live comparison

- Same machine: new ML model, none, LZ4, Zstandard in separate cgroups.
- Match: workload, quota, period, chunk size, repeats.
- Measure: full decision latency, throughput, savings, CPU, prediction cost.
- Next: real datasets; C++ storage integration; more periods/chunk sizes; CPU feature study.
