# ML-guided storage compression under CPU limits

Initial Python implementation of the ML selector and offline evaluation. It trains small decision trees to predict stored size and write latency for no compression, LZ4, and Zstandard, then chooses a codec using a latency budget.

## Quick start

Run from the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m examples.make_demo_data
.venv/bin/python -m compression_ml.selector artifacts/synthetic.csv --budget-us 1000
.venv/bin/python -m unittest discover -s tests -v
```

Code is in `compression_ml/`, the demo generator is in `examples/`, and tests are in `tests/`. Generated data, JSON models, and evaluation results go in `artifacts/` and are Git-ignored.

The demo uses **synthetic measurements** to check the pipeline. Real workload measurements, CPU feature collection, and integration with the C++ storage prototype are still pending.
