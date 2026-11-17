"""Collect a small VM dataset for the compression selector handoff.

Put this file next to collect_measurements.py and run from the repo root:
    .venv/bin/python benchmarks/collect_pilot.py
"""

import csv
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import random
import subprocess
import sys

CODECS = ("none", "lz4", "zstd")
QUOTAS = (25, 50)
PERIOD_US = 100_000
CHUNK_BYTES = 1024 * 1024
CHUNKS = 8
ROUNDS = 2
WORKLOADS = (("repetitive", 0), ("mixed", 2048), ("random", 4096))
FEATURES = ("chunk_bytes", "sample_entropy", "quota_fraction", "period_us")
TARGETS = ("stored_bytes", "write_latency_us")
FIELDS = ("workload_id", "decision_id", "codec", *FEATURES, *TARGETS)


def make_input(path, random_bytes_per_block, rng):
    # Mix within each 4 KiB block so the entropy sample sees both data types.
    pattern = b"ABCDE123"
    repeated = 4096 - random_bytes_per_block

    with path.open("xb") as stream:
        for _ in range(CHUNKS * CHUNK_BYTES // 4096):
            stream.write(pattern * (repeated // len(pattern)))
            if random_bytes_per_block:
                stream.write(rng.randbytes(random_bytes_per_block))


def collect_trial(benchmark, source, workload, quota, codec, destination):
    # A fresh service gives this codec its own quota accounting.
    subprocess.run([
        "systemd-run", "--user", "--wait", "--pipe", "--collect",
        "--expand-environment=no",
        "-p", f"CPUQuota={quota}%",
        "-p", f"CPUQuotaPeriodSec={PERIOD_US // 1000}ms",
        sys.executable, str(benchmark), str(source),
        "--codec", codec, "--chunk-bytes", str(CHUNK_BYTES),
        "--repeats", "1", "--workload-id", workload, "--require-cgroup",
        "--output", str(destination),
    ], check=True)

    metadata = json.loads((destination / "metadata.json").read_text())
    if metadata["status"] != "complete" or metadata["trials"] != CHUNKS:
        raise ValueError(f"Incomplete benchmark: {destination}")

    with (destination / "baseline.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    indices = {int(row["chunk_index"]) for row in rows}
    if len(rows) != CHUNKS or indices != set(range(CHUNKS)):
        raise ValueError(f"Unexpected chunks: {destination}")

    for row in rows:
        if (
            row["workload_id"] != workload
            or row["codec"] != codec
            or row["verified"] != "True"
            or row["cgroup_available"] != "True"
            or int(row["chunk_bytes"]) != CHUNK_BYTES
            or int(row["quota_us"]) != PERIOD_US * quota // 100
            or float(row["quota_fraction"]) != quota / 100
            or int(row["period_us"]) != PERIOD_US
        ):
            raise ValueError(f"Unexpected result or CPU settings: {destination}")

    return rows


def write_csv(path, groups):
    expected = len(WORKLOADS) * len(QUOTAS) * ROUNDS * CHUNKS
    if len(groups) != expected:
        raise ValueError(f"Expected {expected} groups, got {len(groups)}")

    # Joel needs one result for each codec at the same chunk and CPU settings.
    for key, rows in groups.items():
        if len(rows) != len(CODECS) or {row["codec"] for row in rows} != set(CODECS):
            raise ValueError(f"Incomplete codec group: {key}")

        features = {tuple(float(row[name]) for name in FEATURES) for row in rows}
        if len(features) != 1:
            raise ValueError(f"Codec runs have different input features: {key}")

        for row in rows:
            numbers = [float(row[name]) for name in (*FEATURES, *TARGETS)]
            if any(not math.isfinite(value) or value < 0 for value in numbers):
                raise ValueError(f"Invalid measurement: {key}")
            if float(row["sample_entropy"]) > 8 or float(row["write_latency_us"]) <= 0:
                raise ValueError(f"Invalid entropy or latency: {key}")

    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for key in sorted(groups):
            writer.writerows(sorted(groups[key], key=lambda row: row["codec"]))


def main():
    benchmark = Path(__file__).resolve().with_name("collect_measurements.py")
    if not benchmark.is_file():
        raise ValueError("Put collect_pilot.py next to collect_measurements.py")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = Path.cwd() / "artifacts" / f"joel_pilot_{timestamp}"
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(42)
    groups = defaultdict(list)

    for workload, random_bytes in WORKLOADS:
        source = out / f"{workload}.bin"
        make_input(source, random_bytes, rng)

        for quota_index, quota in enumerate(QUOTAS):
            for repeat in range(ROUNDS):
                # Change which codec runs first between repeats and quotas.
                offset = (repeat + quota_index) % len(CODECS)
                order = CODECS[offset:] + CODECS[:offset]

                for codec in order:
                    run = out / f"{workload}_q{quota}_r{repeat}_{codec}"
                    for raw in collect_trial(benchmark, source, workload, quota, codec, run):
                        row = {name: raw[name] for name in FIELDS}
                        row["decision_id"] = f"q{quota}_r{repeat}_c{raw['chunk_index']}"
                        groups[(workload, row["decision_id"])].append(row)

    csv_path = out / "joel_pilot.csv"
    write_csv(csv_path, groups)
    print(f"\nCSV: {csv_path}")
    print(f"{len(groups)} groups; {len(groups) * len(CODECS)} rows")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(f"Pilot collection failed: {error}")
