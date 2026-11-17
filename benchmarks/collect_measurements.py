"""Compression benchmarks and CPU measurements."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

import lz4.frame
import zstandard

# Direct script imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compression_ml.selector import PILOT_FEATURES, choose, load_model, predict

CODECS = ("none", "lz4", "zstd")

# CPU counters (times in microseconds).
STAT_KEYS = (
    "usage_usec",
    "user_usec",
    "system_usec",
    "nr_periods",
    "nr_throttled",
    "throttled_usec",
)

# CSV fields.
CG_FIELDS = (
    "cgroup_available",
    "quota_us",
    "period_us",
    "quota_fraction",
    "cgroup_observation_us",
    *("cgroup_" + key for key in STAT_KEYS),
)

FIELDS = (
    "workload_id",
    "decision_id",
    "repeat",
    "chunk_index",
    "trial_order",
    "codec",
    "chunk_bytes",
    "sample_entropy",
    "stored_bytes",
    "compression_us",
    "write_sync_us",
    "write_latency_us",
    "cpu_time_us",
    "storage_savings_fraction",
    "throughput_mib_s",
    "verified",
    "policy",
    "feature_collection_us",
    "prediction_us",
    "decision_write_latency_us",
    "decision_cpu_time_us",
    "decision_throughput_mib_s",
    "budget_us",
    "budget_missed",
    "outside_training_range",
    *CG_FIELDS,
)


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return value


# First 4 KiB.
def sample_entropy(data):
    sample = data[:4096]
    return -sum(
        (count / len(sample)) * math.log2(count / len(sample))
        for count in Counter(sample).values()
    )


# CPU limits and counters.
class CgroupCPU:
    def __init__(self, directory):
        self.directory = directory
        self.initial_limits = self.limits()
        self.snapshot()

    def limits(self):
        quota, period = (self.directory / "cpu.max").read_text().split()
        period = int(period)
        quota = "max" if quota == "max" else int(quota)

        if period <= 0 or (quota != "max" and quota <= 0):
            raise ValueError("Invalid cpu.max")

        return {
            "quota_us": quota,
            "period_us": period,
            "quota_fraction": "" if quota == "max" else quota / period,
        }

    def snapshot(self):
        lines = (self.directory / "cpu.stat").read_text().splitlines()
        values = {
            key: int(value)
            for key, value in (line.split() for line in lines)
        }

        missing = set(STAT_KEYS) - values.keys()
        if missing:
            raise ValueError(f"Missing cpu.stat counters: {sorted(missing)}")

        return {key: values[key] for key in STAT_KEYS}

    @staticmethod
    def delta(before, after):
        changes = {key: after[key] - before[key] for key in STAT_KEYS}

        if any(value < 0 for value in changes.values()):
            raise ValueError("CPU counters decreased")

        return changes


def detect_cgroup(required):
    try:
        if platform.system() != "Linux":
            raise ValueError("Cgroup CPU monitoring requires Linux")

        # Cgroup v2 path.
        entries = Path("/proc/self/cgroup").read_text().splitlines()
        entry = next((line for line in entries if line.startswith("0::")), None)

        if entry is None:
            raise ValueError("Cgroup v2 not found")

        root = Path("/sys/fs/cgroup").resolve()
        relative = entry.split(":", 2)[2].lstrip("/")
        directory = (root / relative).resolve()

        if not directory.is_relative_to(root):
            raise ValueError("Cgroup path outside mount")

        return CgroupCPU(directory), None

    except (OSError, ValueError) as error:
        if required:
            raise ValueError(f"CPU monitoring unavailable: {error}") from error

        return None, str(error)


def measure(data, codec, destination, compressor, decompressor, monitor,
            decision_start=None, decision_cpu_start=None):
    # Missing measurements stay blank.
    cg_metrics = {key: "" for key in CG_FIELDS}
    cg_metrics["cgroup_available"] = monitor is not None

    if monitor:
        limits = monitor.limits()

        if limits != monitor.initial_limits:
            raise ValueError("CPU quota changed during run")

        cg_metrics.update(limits)
        observation_start = time.perf_counter_ns()
        before = monitor.snapshot()

    # Compression and write timers.
    cpu_start = time.process_time_ns()
    start = time.perf_counter_ns()
    if decision_start is None:
        decision_start = start
    if decision_cpu_start is None:
        decision_cpu_start = cpu_start

    if codec == "none":
        encoded = data
    elif codec == "lz4":
        encoded = lz4.frame.compress(data, compression_level=0)
    else:
        encoded = compressor.compress(data)

    compressed = time.perf_counter_ns()

    # Write and sync.
    with destination.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())

    finished = time.perf_counter_ns()
    cpu_finished = time.process_time_ns()
    cpu_us = (cpu_finished - cpu_start) / 1000
    decision_us = (finished - decision_start) / 1000

    # CPU measurements (before readback).
    if monitor:
        after = monitor.snapshot()
        cg_metrics["cgroup_observation_us"] = (
            time.perf_counter_ns() - observation_start
        ) / 1000
        cg_metrics.update({
            "cgroup_" + key: value
            for key, value in monitor.delta(before, after).items()
        })

        if monitor.limits() != limits:
            raise ValueError("CPU quota changed during trial")

    # Readback check.
    saved = destination.read_bytes()

    if codec == "none":
        restored = saved
    elif codec == "lz4":
        restored = lz4.frame.decompress(saved)
    else:
        restored = decompressor.decompress(saved)

    if restored != data:
        raise RuntimeError(f"Verification failed: {destination}")

    latency_us = (finished - start) / 1000

    return {
        # File size in bytes.
        "stored_bytes": len(saved),
        "compression_us": (compressed - start) / 1000,
        "write_sync_us": (finished - compressed) / 1000,
        "write_latency_us": latency_us,
        "cpu_time_us": cpu_us,
        "storage_savings_fraction": 1 - len(saved) / len(data),
        "throughput_mib_s": len(data) / 2**20 / (latency_us / 1e6),
        "verified": True,
        "decision_write_latency_us": decision_us,
        "decision_cpu_time_us": (cpu_finished - decision_cpu_start) / 1000,
        "decision_throughput_mib_s": len(data) / 2**20 / (decision_us / 1e6),
        **cg_metrics,
    }


def run(args):
    # Model.
    model = None
    if args.codec == "ml":
        if args.model is None or args.budget_us is None:
            raise ValueError("--codec ml requires --model and --budget-us")
        if not math.isfinite(args.budget_us) or args.budget_us <= 0:
            raise ValueError("--budget-us must be positive and finite")
        model = load_model(args.model)
        if tuple(model["features"]) != PILOT_FEATURES:
            raise ValueError("Four-feature pilot model required")
    elif args.model is not None or args.budget_us is not None:
        raise ValueError("Use --codec ml with --model and --budget-us")
    codecs = CODECS if args.codec in ("all", "ml") else (args.codec,)
    source = args.input.resolve()

    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("Nonempty input file required")

    monitor, reason = detect_cgroup(args.require_cgroup or model is not None)
    if model is not None and monitor.initial_limits["quota_us"] == "max":
        raise ValueError("ML mode requires a finite CPU quota")

    if monitor:
        print(f"Cgroup: {monitor.directory}")
        limits = monitor.initial_limits
        print(f"cpu.max: {limits['quota_us']} {limits['period_us']}")
    else:
        print(f"CPU stats: unavailable ({reason}); CSV fields blank")

    # Output directory (must be new).
    args.output.mkdir(parents=True, exist_ok=False)
    scratch = args.output / "trial_files"
    scratch.mkdir()

    compressor = zstandard.ZstdCompressor(level=3, threads=0)
    decompressor = zstandard.ZstdDecompressor()
    rng = random.Random(args.seed)

    # Run settings.
    metadata = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            "lz4": version("lz4"),
            "zstandard": version("zstandard"),
        },
        "input": str(source),
        "input_bytes": source.stat().st_size,
        "chunk_bytes": args.chunk_bytes,
        "repeats": args.repeats,
        "codecs": list(codecs),
        "policy": args.codec,
        "model": str(args.model.resolve()) if model is not None else None,
        "model_training": model.get("training") if model is not None else None,
        "budget_us": args.budget_us,
        "seed": args.seed,
        "lz4_level": 0,
        "zstd_level": 3,
        "zstd_worker_threads": 0,
        "cgroup": {
            "available": monitor is not None,
            "unavailable_reason": reason,
            "limits": monitor.initial_limits if monitor else None,
        },
    }

    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    totals = {
        codec: {"input": 0, "stored": 0, "us": 0, "decision_us": 0,
                "prediction_us": 0, "budget_misses": 0, "count": 0, "throttled_us": 0}
        for codec in codecs
    }

    # Whole-run timers.
    run_before = monitor.snapshot() if monitor else None
    run_start = time.perf_counter_ns()

    try:
        with (args.output / "baseline.csv").open("x", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
            writer.writeheader()

            for repeat in range(args.repeats):
                with source.open("rb") as stream:
                    chunk_index = 0

                    while True:
                        data = stream.read(args.chunk_bytes)
                        if not data:
                            break

                        order = ["ml"] if model is not None else list(codecs)
                        # Shared quota; shuffle codec order.
                        if args.codec == "all":
                            rng.shuffle(order)

                        for position, codec in enumerate(order):
                            decision_cpu_start = time.process_time_ns()
                            decision_start = time.perf_counter_ns()
                            entropy = sample_entropy(data)
                            context = {"chunk_bytes": len(data), "sample_entropy": entropy}
                            if model is not None:
                                limits = monitor.limits()
                                context["quota_fraction"] = limits["quota_fraction"]
                                context["period_us"] = limits["period_us"]
                            features_finished = time.perf_counter_ns()
                            prediction_us = 0
                            outside_range = ""
                            if model is not None:
                                codec = choose(predict(model, context), args.budget_us)
                                prediction_us = (time.perf_counter_ns() - features_finished) / 1000
                                ranges = model.get("training", {}).get("training_ranges", {})
                                outside_range = False
                                for feature, (low, high) in ranges.items():
                                    if context[feature] < low or context[feature] > high:
                                        outside_range = True
                            destination = scratch / f"r{repeat}_c{chunk_index}_{codec}.bin"
                            metrics = measure(
                                data, codec, destination,
                                compressor, decompressor, monitor,
                                decision_start, decision_cpu_start,
                            )
                            budget_missed = (
                                metrics["decision_write_latency_us"] > args.budget_us
                                if model is not None else ""
                            )

                            # CSV row.
                            writer.writerow({
                                "workload_id": args.workload_id or source.name,
                                "decision_id": f"r{repeat}_c{chunk_index}",
                                "repeat": repeat,
                                "chunk_index": chunk_index,
                                "trial_order": position,
                                "codec": codec,
                                "chunk_bytes": len(data),
                                "sample_entropy": entropy,
                                "policy": args.codec,
                                "feature_collection_us": (features_finished - decision_start) / 1000,
                                "prediction_us": prediction_us,
                                "budget_us": args.budget_us if model is not None else "",
                                "budget_missed": budget_missed,
                                "outside_training_range": outside_range,
                                **metrics,
                            })

                            total = totals[codec]
                            total["input"] += len(data)
                            total["stored"] += metrics["stored_bytes"]
                            total["us"] += metrics["write_latency_us"]
                            total["decision_us"] += metrics["decision_write_latency_us"]
                            total["prediction_us"] += prediction_us
                            total["budget_misses"] += int(bool(budget_missed))
                            total["count"] += 1

                            if monitor:
                                total["throttled_us"] += metrics["cgroup_throttled_usec"]

                            destination.unlink()

                        chunk_index += 1

                csv_file.flush()

        if monitor:
            metadata["run_cgroup_delta"] = monitor.delta(run_before, monitor.snapshot())
            metadata["run_cgroup_scope"] = (
                "Whole loop: reads, features, selection, writes, verification, logging, cleanup"
            )

        metadata["run_elapsed_seconds"] = (time.perf_counter_ns() - run_start) / 1e9
        scratch.rmdir()
        metadata["status"] = "complete"
        metadata["trials"] = sum(total["count"] for total in totals.values())
        metadata["codec_counts"] = {codec: total["count"] for codec, total in totals.items()}
        total_input = sum(t["input"] for t in totals.values())
        total_decision_us = sum(t["decision_us"] for t in totals.values())
        prediction_us = sum(total["prediction_us"] for total in totals.values())
        misses = sum(total["budget_misses"] for total in totals.values())
        metadata["decision_throughput_mib_s"] = total_input / 2**20 / (total_decision_us / 1e6)
        metadata["mean_prediction_us"] = prediction_us / metadata["trials"]
        metadata["budget_miss_fraction"] = (
            misses / metadata["trials"] if model is not None else None
        )

    except (Exception, KeyboardInterrupt) as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise

    finally:
        # Final status.
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    for codec, total in totals.items():
        if total["count"] == 0:
            continue
        savings = 1 - total["stored"] / total["input"]
        mean_latency = total["us"] / total["count"]
        throughput = total["input"] / 2**20 / (total["us"] / 1e6)

        print(
            f"{codec:4} | trials: {total['count']} | "
            f"savings: {savings:.1%} | write_us: {mean_latency:.1f} | "
            f"MiB/s: {throughput:.2f}"
        )

        if monitor:
            print(f"  Throttled_us: {total['throttled_us']}")

    if monitor:
        print("Run CPU counters:", metadata["run_cgroup_delta"])

    print("Readback: passed")
    print(f"Results: {args.output / 'baseline.csv'}")
    print(f"Decision MiB/s: {metadata['decision_throughput_mib_s']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--chunk-bytes", type=positive_int, default=262144)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workload-id")
    parser.add_argument(
        "--codec", choices=("all", "ml", *CODECS), default="all",
        help="Fixed codec, all codecs, or ML",
    )
    parser.add_argument("--model", type=Path, help="ML model JSON")
    parser.add_argument("--budget-us", type=float, help="ML write budget (microseconds)")
    parser.add_argument(
        "--require-cgroup",
        action="store_true",
        help="Require CPU limits and counters",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / datetime.now().strftime(
            "baseline_%Y%m%d_%H%M%S_%f"
        ),
    )

    args = parser.parse_args()

    try:
        run(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
