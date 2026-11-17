# Pilot

- Data: `data/compression_pilot.csv` — 288 VM measurements.
- Inputs: chunk bytes, sample entropy, CPU quota fraction, period (µs).
- Targets: stored bytes, write latency (µs).
- Training: random + repetitive — 64 decisions.
- Test: mixed — 32 decisions, both quotas, all repeats.
- Chunk size: 1 MiB. Period: 100 ms. Quotas: 25%, 50%.
- Model: `models/pilot/model.json`.
- Results: `models/pilot/evaluation.json`.

## Train

```bash
.venv/bin/python -m compression_ml.selector data/compression_pilot.csv \
  --feature-set pilot --holdout-workload mixed --budget-us 1000 \
  --output artifacts/pilot_check
```

## Call

Load the model once. Call before compressing each chunk.

```python
from compression_ml.selector import load_model, predict, choose

model = load_model("models/pilot/model.json")
inputs = {
    "chunk_bytes": len(data),
    "sample_entropy": entropy,  # First 4 KiB.
    "quota_fraction": quota_us / period_us,
    "period_us": period_us,
}
codec = choose(predict(model, inputs), budget_us=1000)
```

Choice: smallest predicted size within the write budget; otherwise fastest.
Inference: Python standard library only.

## Benchmark

Linux VM. Replace the input path. Use a new output directory per run.

```bash
systemd-run --user --wait --pipe --collect \
  -p CPUQuota=25% -p CPUQuotaPeriodSec=100ms \
  --working-directory="$PWD" \
  "$PWD/.venv/bin/python" "$PWD/benchmarks/collect_measurements.py" \
  /absolute/path/workload.bin \
  --codec ml --model "$PWD/models/pilot/model.json" --budget-us 1000 \
  --chunk-bytes 1048576 --repeats 2 --workload-id mixed \
  --require-cgroup --output "$PWD/artifacts/ml_q25"
```

- Second quota: `CPUQuota=50%`, new output directory.
- Baselines: replace `--codec ml --model ... --budget-us 1000` with `--codec none`, `--codec lz4`, or `--codec zstd`.
- Comparisons: same input, chunk size, quotas, periods, and repeats.
- `write_latency_us`: compression through fsync.
- `decision_write_latency_us`: features + selection + compression + fsync.
- `prediction_us`: prediction and codec choice.
- `budget_missed`: decision latency exceeds budget.
- `outside_training_range`: input outside training bounds.
- Readback and logging: outside latency timers.

## Status

- Budget: 1,000 µs example.
- Held-out choices: no compression, all 32 decisions.
- Storage savings: 0%. Write-budget misses: 25% (offline replay).
- Improvement over no compression: none in this test.
- Local tests: codec routing and readback with simulated cgroup readings.
- Pending: live VM comparisons under CPU limits; C++ storage integration.

Pilot coverage: three generated workloads, one chunk size, one period. Replay excludes live selector costs. Predicted budgets are not guarantees.
