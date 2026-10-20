# ML-guided storage compression under CPU limits

Joel's initial ML and evaluation pipeline. Gus's C++ storage prototype and real benchmark data are not present yet.

## Project layout

```text
ml-storage-compression/
├── compression_ml/        # Training, selection, and offline evaluation
│   └── selector.py
├── examples/              # Synthetic demo data generator
│   └── make_demo_data.py
├── tests/                 # Automated checks
│   └── test_selector.py
├── artifacts/             # Generated datasets, models, and results (Git-ignored)
├── requirements.txt
└── README.md
```

Run the commands below from this repository's root directory. The local `.venv/` is also Git-ignored.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m examples.make_demo_data
.venv/bin/python -m compression_ml.selector artifacts/synthetic.csv --budget-us 1000
.venv/bin/python -m unittest discover -s tests -v
```

The demo uses **invented synthetic measurements** to verify software, not research results. The 1,000 µs budget is an example, not an experimentally established target.

Training creates `artifacts/model.json`, `ablation_model.json`, and `evaluation.json`. Separate depth-limited trees predict stored bytes and write latency per codec. The selector minimizes predicted size among codecs meeting the budget; if none meets it, it selects the predicted fastest codec. Predictions do not guarantee latency.

Baselines: fixed none/LZ4/Zstandard and ML without quota, period, or recent throttling (CPU utilization is retained). A deterministic split holds out 25% of workload IDs. Use the same workload ID for related runs from the same source dataset, including across CPU settings. Final evaluation needs more workloads, repeated trials, uncertainty estimates, and separate validation data for tuning.

## Shared measurement format

```csv
workload_id,decision_id,codec,chunk_bytes,sample_entropy,cpu_utilization,quota_fraction,period_us,recent_throttle_fraction,stored_bytes,write_latency_us
```

Each `(workload_id, decision_id)` requires exactly three rows: `none`, `lz4`, `zstd`, with identical pre-decision features. Replay the same chunk and intended CPU conditions separately for each codec. Randomize codec order and repeat measurements. Distinguish trials with decision IDs; record environment and settings in a companion benchmark manifest.

| Column | Meaning |
| --- | --- |
| workload_id | Source workload/dataset identity for train/test separation |
| decision_id | Chunk/context/trial identity within workload |
| codec | `none`, `lz4`, `zstd`; fix codec levels and record in manifest |
| chunk_bytes | Original size in bytes, positive |
| sample_entropy | Shannon byte entropy, 0–8 bits/byte, sampled before compression; record sampling policy |
| cpu_utilization | CPU usage / wall time / allocated logical CPUs, 0–1, over a preceding observation window |
| quota_fraction | Quota µs / period µs in CPU equivalents; 0 means unlimited, values above 1 allowed |
| period_us | Enforcement period in µs, positive, even with unlimited quota |
| recent_throttle_fraction | Delta nr_throttled / delta nr_periods over preceding window, 0–1; 0 if no periods elapsed |
| stored_bytes | Measured size including chunk metadata, nonnegative |
| write_latency_us | Measured compression-plus-write latency, positive; define write/fsync semantics in manifest |

Features must exist **before** selection; post-compression sizes/timings are labels only. Matching features across three independent codec runs is an offline approximation, since live scheduler state cannot be identical. CPU feature collection remains to be implemented and checked on the shared VM.

## Integration

Exported JSON contains tree arrays. After training, Python inference needs only the standard library:

```python
import json
from compression_ml.selector import predict, choose

model = json.load(open("artifacts/model.json"))  # Load once at startup.
# context holds the six pre-decision features above.
codec = choose(predict(model, context), budget_us=1000)
```

This callable is the initial integration boundary; C++ integration is pending. Gus can implement the JSON tree traversal, or we can add a persistent interface once the storage API is agreed. Starting Python per chunk would add overhead that must be measured.

## Evaluation limits and next work

The report records train/test workloads, storage savings, mean/p95 write latency, budget misses, codec counts, serial replay throughput, and local prediction-plus-selection time. Replay throughput is original bytes / summed selected latencies, not measured concurrent throughput. Prediction timing excludes feature collection, JSON loading, IPC, and storage integration; it is separate from recorded write latency. Fixed-codec prediction overhead is zero by definition.

Next: collect actual storage measurements with readback checks; train on real data; integrate feature collection and inference under the same CPU cgroup as compression; measure end-to-end latency, CPU usage, throughput, and selector overhead. Compare equal quota/period ratios at different periods. Offline replay does not establish live performance or CPU savings.

Reference: [scikit-learn DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html).
