# Experiments

## Record

- History: `experiments/runs.csv` — one row per policy.
- Saved: settings, workload split, data/model hashes, savings, latency, misses, overhead.
- Output: new directory per run; existing logged models stay unchanged.
- Budget: 1,000 µs provisional. Agree on the target before final testing.

## 01 — Pilot

- Model: `models/pilot/model.json`.
- Data: 288 VM measurements; three generated workloads.
- Training: random + repetitive. Test: mixed.
- Result: 0% savings; 25% write-budget misses (offline replay).
- Finding: mixed-data compressed size was overpredicted.
- Status: baseline saved; original results imported into the history.

## 02 — More data

- First: run the pilot ML benchmark on the VM. Command: [Pilot](../docs/PILOT.md).
- Coverage: nine mixes, from repetitive to random.
- Repeats: five. Chunks: eight per workload, 1 MiB each.
- CPU: 25% and 50%, 100 ms period.
- Expected: 2,160 measured codec rows.
- Validation: `mix_3072`. Final test: `mix_1536`.
- Files: `development.csv`, `test.csv`, raw trials, `collection.json`.
- Status: collection pending on the same VM.

Run from the repository root on the VM:

```bash
.venv/bin/python benchmarks/collect_pilot.py --expanded --output artifacts/expanded_01
```

Require `collection.json` status `complete`. Timing spikes stay in the data.

## 03 — Validation

```bash
.venv/bin/python -m compression_ml.selector artifacts/expanded_01/development.csv \
  --feature-set pilot --holdout-workload mix_3072 --role validation \
  --budget-us 1000 --output models/v2_validation
```

- Training: seven mixes. Validation: one entire mix, all quotas and repeats.
- Compare: size/time prediction errors, savings, budget misses, fixed codecs.
- Tune if needed: `--max-depth`, `--min-samples-leaf`; new output per attempt.
- Final test: keep `test.csv` unused during tuning.

## 04 — Final test

After choosing settings using validation:

```bash
.venv/bin/python -m compression_ml.selector artifacts/expanded_01/development.csv \
  --feature-set pilot --test-data artifacts/expanded_01/test.csv \
  --budget-us 1000 --output models/v2_final
```

- Use the chosen tree settings; defaults are depth 5, leaf size 2.
- Training: all development workloads. Test: reserved workload.
- One final evaluation; further tuning needs a new reserved test set.
- These are held-out generated mix levels, not independent real datasets.

## 05 — Live comparison

- Same VM: ML, none, LZ4, Zstandard in separate cgroups.
- Match: workload, quota, period, chunk size, repeats.
- Measure: full decision latency, throughput, savings, CPU, prediction cost.
- Next: real datasets; C++ storage integration; more periods/chunk sizes; CPU feature study.
