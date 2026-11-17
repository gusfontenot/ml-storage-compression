"""Collect pilot measurements on the VM."""

import argparse
import csv
from collections import defaultdict
from datetime import datetime
import json
import hashlib
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
EXPANDED_WORKLOADS = (
    ("repetitive", 0), ("mix_512", 512), ("mix_1024", 1024),
    ("mix_1536", 1536), ("mixed", 2048), ("mix_2560", 2560),
    ("mix_3072", 3072), ("mix_3584", 3584), ("random", 4096),
)
VALIDATION_WORKLOAD = "mix_3072"
TEST_WORKLOAD = "mix_1536"
FEATURES = ("chunk_bytes", "sample_entropy", "quota_fraction", "period_us")
TARGETS = ("stored_bytes", "write_latency_us")
FIELDS = ("workload_id", "decision_id", "codec", *FEATURES, *TARGETS)


def make_input(path, random_bytes_per_block, rng):
    # Mixed 4 KiB blocks.
    pattern = b"ABCDE123"
    repeated = 4096 - random_bytes_per_block

    with path.open("xb") as stream:
        for _ in range(CHUNKS * CHUNK_BYTES // 4096):
            stream.write(pattern * (repeated // len(pattern)))
            if random_bytes_per_block:
                stream.write(rng.randbytes(random_bytes_per_block))


def collect_trial(benchmark, source, workload, quota, codec, destination):
    # Separate cgroup per codec.
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


def write_csv(path, groups, expected=None):
    if expected is None:
        expected = len(WORKLOADS) * len(QUOTAS) * ROUNDS * CHUNKS
    if len(groups) != expected:
        raise ValueError(f"Expected {expected} groups, got {len(groups)}")

    # Three codecs per decision.
    for key, rows in groups.items():
        if len(rows) != len(CODECS) or {row["codec"] for row in rows} != set(CODECS):
            raise ValueError(f"Incomplete codec group: {key}")

        features = {tuple(float(row[name]) for name in FEATURES) for row in rows}
        if len(features) != 1:
            raise ValueError(f"Mismatched inputs: {key}")

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


def collect(args):
    benchmark = Path(__file__).resolve().with_name("collect_measurements.py")
    if not benchmark.is_file():
        raise ValueError("Missing collect_measurements.py")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    workloads = EXPANDED_WORKLOADS if args.expanded else WORKLOADS
    rounds = args.rounds
    if rounds is None:
        rounds = 5 if args.expanded else ROUNDS
    if rounds < 1:
        raise ValueError("Rounds must be positive")
    prefix = "expanded" if args.expanded else "joel_pilot"
    out = args.output or Path.cwd() / "artifacts" / f"{prefix}_{timestamp}"
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(42)
    groups = defaultdict(list)
    settings = {
        "status": "incomplete", "seed": 42, "rounds": rounds,
        "workloads": dict(workloads), "quotas": list(QUOTAS),
        "period_us": PERIOD_US, "chunk_bytes": CHUNK_BYTES, "chunks": CHUNKS,
        "validation_workload": VALIDATION_WORKLOAD if args.expanded else None,
        "test_workload": TEST_WORKLOAD if args.expanded else None,
        "benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {},
    }
    manifest = out / "collection.json"
    manifest.write_text(json.dumps(settings, indent=2) + "\n")

    for workload, random_bytes in workloads:
        source = out / f"{workload}.bin"
        make_input(source, random_bytes, rng)
        settings["input_sha256"][workload] = hashlib.sha256(source.read_bytes()).hexdigest()

        for quota_index, quota in enumerate(QUOTAS):
            for repeat in range(rounds):
                # Rotate codec order.
                offset = (repeat + quota_index) % len(CODECS)
                order = CODECS[offset:] + CODECS[:offset]

                for codec in order:
                    run = out / f"{workload}_q{quota}_r{repeat}_{codec}"
                    for raw in collect_trial(benchmark, source, workload, quota, codec, run):
                        row = {name: raw[name] for name in FIELDS}
                        row["decision_id"] = f"q{quota}_r{repeat}_c{raw['chunk_index']}"
                        groups[(workload, row["decision_id"])].append(row)

    csv_path = out / "joel_pilot.csv"
    per_workload = len(QUOTAS) * rounds * CHUNKS
    write_csv(csv_path, groups, len(workloads) * per_workload)
    if args.expanded:
        development = {key: rows for key, rows in groups.items() if key[0] != TEST_WORKLOAD}
        testing = {key: rows for key, rows in groups.items() if key[0] == TEST_WORKLOAD}
        write_csv(out / "development.csv", development, (len(workloads) - 1) * per_workload)
        write_csv(out / "test.csv", testing, per_workload)
        print(f"Development: {out / 'development.csv'}")
        print(f"Final test: {out / 'test.csv'}")
    settings["status"] = "complete"
    settings["rows"] = len(groups) * len(CODECS)
    manifest.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"\nCSV: {csv_path}")
    print(f"Decisions: {len(groups)} | Rows: {len(groups) * len(CODECS)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded", action="store_true", help="Nine workload levels")
    parser.add_argument("--rounds", type=int, help="Repeats: pilot 2, expanded 5")
    parser.add_argument("--output", type=Path, help="New output directory")
    collect(parser.parse_args())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(f"Pilot failed: {error}")
