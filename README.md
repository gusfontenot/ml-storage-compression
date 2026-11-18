# ML storage compression

Python selector: no compression, LZ4, or Zstandard.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Demo

```bash
.venv/bin/python -m examples.make_demo_data
.venv/bin/python -m compression_ml.selector artifacts/synthetic.csv --budget-us 1000
```

Demo data: generated values.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Pilot model and benchmark commands: [Pilot](docs/PILOT.md).

Current measurements: [Results](docs/RESULTS.md).

Next runs and history: [Experiments](experiments/README.md).
