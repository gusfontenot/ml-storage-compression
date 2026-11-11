"""Compare compression methods and record Linux CPU quota stats.

Run this from the repo root:
    python benchmarks/collect_measurements.py README.md

CPU limits are set externally using systemd-run.
This script writes raw benchmark measurements.
"""

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
import time

import lz4.frame
import zstandard

# Keeping the uncompressed case gives us a baseline for size and write time.
CODECS = ("none", "lz4", "zstd")

# Counters from cpu.stat. The *_usec values are in microseconds.
STAT_KEYS = (
    "usage_usec",
    "user_usec",
    "system_usec",
    "nr_periods",
    "nr_throttled",
    "throttled_usec",
)

# Add the CPU measurements to each trial row.
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
    *CG_FIELDS,
)


# A zero chunk size would look like end of file, and zero repeats means no trials.
def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return value


# Sample the start of the chunk to keep the entropy check cheap on large inputs.
def sample_entropy(data):
    sample = data[:4096]
    return -sum(
        (count / len(sample)) * math.log2(count / len(sample))
        for count in Counter(sample).values()
    )


# Read CPU limits and counters for the current cgroup.
class CgroupCPU:
    def __init__(self, directory):
        # Save the starting limits so we can tell if the quota changes.
        self.directory = directory
        self.initial_limits = self.limits()
        # Check that the counters are readable before we start.
        self.snapshot()

    def limits(self):
        # cpu.max gives the allowed CPU time and the length of each period.
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
        # Counters are cumulative, so we subtract the starting readings.
        changes = {key: after[key] - before[key] for key in STAT_KEYS}

        if any(value < 0 for value in changes.values()):
            raise ValueError("The cgroup counters decreased, so discard the run")

        return changes


def detect_cgroup(required):
    try:
        if platform.system() != "Linux":
            raise ValueError("Cgroup CPU monitoring requires Linux")

        # Find the cgroup this process is in. The v2 line starts with 0::.
        entries = Path("/proc/self/cgroup").read_text().splitlines()
        entry = next((line for line in entries if line.startswith("0::")), None)

        if entry is None:
            raise ValueError("There was no cgroup v2 membership found")

        root = Path("/sys/fs/cgroup").resolve()
        relative = entry.split(":", 2)[2].lstrip("/")
        directory = (root / relative).resolve()

        if not directory.is_relative_to(root):
            raise ValueError("The cgroup path is outside the supported mount")

        return CgroupCPU(directory), None

    except (OSError, ValueError) as error:
        # Fail for quota experiments, but allow ordinary runs without Linux stats.
        if required:
            raise ValueError(
                f"The required cgroup CPU monitoring is unavailable: {error}"
            ) from error

        return None, str(error)


def measure(data, codec, destination, compressor, decompressor, monitor):
    # Leave unavailable measurements blank so they aren't mistaken for zero.
    cg_metrics = {key: "" for key in CG_FIELDS}
    cg_metrics["cgroup_available"] = monitor is not None

    if monitor:
        limits = monitor.limits()

        if limits != monitor.initial_limits:
            raise ValueError("The CPU quota changed during the run, so discard this run")

        cg_metrics.update(limits)
        observation_start = time.perf_counter_ns()
        before = monitor.snapshot()

    # CPU time measures work done by this process. Elapsed time includes waits.
    cpu_start = time.process_time_ns()
    start = time.perf_counter_ns()

    if codec == "none":
        encoded = data
    elif codec == "lz4":
        encoded = lz4.frame.compress(data, compression_level=0)
    else:
        encoded = compressor.compress(data)

    compressed = time.perf_counter_ns()

    # Flush the Python buffer and sync the file the same way for every codec.
    with destination.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())

    finished = time.perf_counter_ns()
    cpu_us = (time.process_time_ns() - cpu_start) / 1000

    # Take the second cgroup reading before readback to leave verification out.
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
            raise ValueError("The CPU quota changed during the trial, so discard this run")

    # Verify the file that was written, not just the buffer in memory.
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
        # Count file bytes, not the number of disk blocks.
        "stored_bytes": len(saved),
        "compression_us": (compressed - start) / 1000,
        "write_sync_us": (finished - compressed) / 1000,
        "write_latency_us": latency_us,
        "cpu_time_us": cpu_us,
        "storage_savings_fraction": 1 - len(saved) / len(data),
        "throughput_mib_s": len(data) / 2**20 / (latency_us / 1e6),
        "verified": True,
        **cg_metrics,
    }


def run(args):
    # A single codec gets its own run and can be given its own cgroup.
    codecs = CODECS if args.codec == "all" else (args.codec,)
    source = args.input.resolve()

    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("Input must be a nonempty regular file")

    monitor, reason = detect_cgroup(args.require_cgroup)

    if monitor:
        print(f"Cgroup: {monitor.directory}")
        limits = monitor.initial_limits
        print(f"cpu.max: {limits['quota_us']} {limits['period_us']}")
    else:
        print(f"Cgroup monitoring unavailable; CPU-control CSV fields are blank: {reason}")

    # Require a new output directory so old results aren't overwritten.
    args.output.mkdir(parents=True, exist_ok=False)
    scratch = args.output / "trial_files"
    scratch.mkdir()

    compressor = zstandard.ZstdCompressor(level=3, threads=0)
    decompressor = zstandard.ZstdDecompressor()
    rng = random.Random(args.seed)

    # Save the settings we need to repeat this run.
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
        codec: {"input": 0, "stored": 0, "us": 0, "count": 0, "throttled_us": 0}
        for codec in codecs
    }

    # These readings cover the entire loop, including work between timed writes.
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

                        entropy = sample_entropy(data)
                        order = list(codecs)
                        # In all-codec mode, shuffle so one codec isn't always first.
                        # These trials still share a quota, so order can affect timing.
                        if args.codec == "all":
                            rng.shuffle(order)

                        for position, codec in enumerate(order):
                            destination = scratch / f"r{repeat}_c{chunk_index}_{codec}.bin"
                            metrics = measure(
                                data, codec, destination,
                                compressor, decompressor, monitor,
                            )

                            # The chunk and repeat ID stays the same across codecs.
                            writer.writerow({
                                "workload_id": args.workload_id or source.name,
                                "decision_id": f"r{repeat}_c{chunk_index}",
                                "repeat": repeat,
                                "chunk_index": chunk_index,
                                "trial_order": position,
                                "codec": codec,
                                "chunk_bytes": len(data),
                                "sample_entropy": entropy,
                                **metrics,
                            })

                            total = totals[codec]
                            total["input"] += len(data)
                            total["stored"] += metrics["stored_bytes"]
                            total["us"] += metrics["write_latency_us"]
                            total["count"] += 1

                            if monitor:
                                total["throttled_us"] += metrics["cgroup_throttled_usec"]

                            # Remove each file after it is verified.
                            destination.unlink()

                        chunk_index += 1

                csv_file.flush()

        if monitor:
            metadata["run_cgroup_delta"] = monitor.delta(run_before, monitor.snapshot())
            metadata["run_cgroup_scope"] = (
                "Entire benchmark loop, including input reads, entropy, "
                "CSV logging, verification and cleanup"
            )

        metadata["run_elapsed_seconds"] = (time.perf_counter_ns() - run_start) / 1e9
        scratch.rmdir()
        metadata["status"] = "complete"
        metadata["trials"] = sum(total["count"] for total in totals.values())

    except (Exception, KeyboardInterrupt) as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise

    finally:
        # Save the final status even if a trial fails.
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    for codec, total in totals.items():
        savings = 1 - total["stored"] / total["input"]
        mean_latency = total["us"] / total["count"]
        # Divide the input size by the time spent compressing and writing.
        throughput = total["input"] / 2**20 / (total["us"] / 1e6)

        print(
            f"{codec:4} | {total['count']} trials | "
            f"savings {savings:.1%} | "
            f"mean latency {mean_latency:.1f} us | "
            f"throughput {throughput:.2f} MiB/s"
        )

        if monitor:
            print(
                "     The summed trial cgroup throttled time: "
                f"{total['throttled_us']} us"
            )

    if monitor:
        print("The whole loop cgroup counters:", metadata["run_cgroup_delta"])

    print(f"All of the outputs were verified. Results: {args.output / 'baseline.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--chunk-bytes", type=positive_int, default=262144)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workload-id")
    parser.add_argument(
        "--codec", choices=("all", *CODECS), default="all",
        help="Run one fixed codec, or all three in the same run (default).",
    )
    parser.add_argument(
        "--require-cgroup",
        action="store_true",
        help="Fail if cgroup CPU limits and counters cannot be read",
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
