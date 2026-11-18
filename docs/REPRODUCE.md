- Requirements: Git, Python 3.12, venv support. Tested on Ubuntu 24.04 / WSL2.
- Saved-data checks: no CPU quota setup needed.
- New benchmarks: Linux, systemd user services, cgroup v2 CPU quotas.
- Run commands from the repository root in a regular terminal.

- Setup on Ubuntu 24.04:

```bash
sudo apt update
sudo apt install git python3-venv
git clone https://github.com/gusfontenot/ml-storage-compression.git
cd ml-storage-compression
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

- Repository access: authenticate with GitHub when cloning if needed.
- Existing checkout: run `git pull --ff-only` first.
- Locked packages: the versions used for the recorded results.

- Reproduce the saved results:

```bash
.venv/bin/python -m examples.reproduce_results
```

- Data: saved development CSV. Seed: 42. Tree: depth 5, leaf size 2.
- Split: 560 training decisions, 80 validation decisions; whole workloads stay together.
- Checks: data hash, savings, write latency, throughput, codec choices, prediction errors.
- Expected: `Saved results: matched`.
- Output: a new `artifacts/reproduce_*` directory, including a local run log.
- Original models/results: kept unchanged. Reserved final test: unused.
- Prediction timing: measured again; it will vary.

- Run tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

- Repeat data collection on Linux:

```bash
systemd-run --user --wait --pipe --collect -p CPUQuota=25% /bin/true
.venv/bin/python benchmarks/collect_pilot.py --expanded --output artifacts/repeat_01
```

- Expected: 2,160 codec measurements; `collection.json` status `complete`.
- Settings: nine workload levels, five repeats, eight chunks, 1 MiB, 25%/50% CPU quotas, 100 ms period.
- Outputs: raw trials, generated input files, development/test CSVs, settings, input hashes.
- Repeats: use a new output directory each time.
- WSL2: systemd must be enabled. Containers/restricted sessions may not support user services.
- Bus or CPU-controller errors: use a Linux login session with systemd and working CPU quotas.

- Repeat the original live integration check:

```bash
.venv/bin/python -m examples.reproduce_live
```

- Expected: 128 verified writes, four policies, both quotas.
- Input: regenerated with seed 42 and checked against the saved input hash.
- Model: original `models/pilot/model.json`, matching the recorded live check.
- Outputs: a new `artifacts/live_*` directory with raw CSVs, metadata, and a console log.
- Exact live latencies: vary by machine and run.

- Train on the new measurements:

```bash
.venv/bin/python -m compression_ml.selector artifacts/repeat_01/development.csv \
  --feature-set pilot --holdout-workload mix_3072 --role validation \
  --budget-us 1000 --output artifacts/repeat_model_01 \
  --log artifacts/repeat_model_01/runs.csv
```

- Budget: 1,000 µs example, matching the recorded experiment.
- Final test: keep `test.csv` out of training and tuning.
- New timings: depend on CPU, disk, OS, load, and throttling; they will differ from saved measurements.
- Model-driven live runs: [Pilot commands](PILOT.md). After collection, use `artifacts/repeat_01/mix_3072.bin` as the input and the desired saved model.
- Published data/models: included in the repo. Large raw trial directories: regenerated under Git-ignored `artifacts/`.
