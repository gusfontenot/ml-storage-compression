"""Development baseline: chunk compression + file write/flush/fsync.

Run from the repository root:
    python benchmarks/collect_measurements.py README.md

The CSV intentionally omits Linux CPU features; it is NOT yet selector input.
Input reads, entropy sampling, decompression checks and cleanup are untimed.
Each trial creates a separate file. fsync is included, but directory syncing
and platform-specific power-loss guarantees are outside this prototype.
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

CODECS = ("none", "lz4", "zstd")
FIELDS = (
    "workload_id", "decision_id", "repeat", "chunk_index", "trial_order",
    "codec", "chunk_bytes", "sample_entropy", "stored_bytes",
    "compression_us", "write_sync_us", "write_latency_us", "cpu_time_us",
    "storage_savings_fraction", "throughput_mib_s", "verified",
)

def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return value

def sample_entropy(data):
    """Shannon byte entropy of the first at most 4096 bytes, bits/byte."""
    sample = data[:4096]
    return -sum((n / len(sample)) * math.log2(n / len(sample))
                for n in Counter(sample).values())    

def measure(data, codec, destination, compressor, decompressor):
    cpu_start = time.process_time_ns()
    start = time.perf_counter_ns()
    if codec == "none":
        encoded = data
    elif codec == "lz4":
        encoded = lz4.frame.compress(data, compression_level=0)
    else:
        encoded = compressor.compress(data)
    compressed = time.perf_counter_ns()
    with destination.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    finished = time.perf_counter_ns()
    cpu_us = (time.process_time_ns() - cpu_start) / 1000

    # Read back the actual output, not the in-memory compressed buffer.
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
        "stored_bytes": len(saved),
        "compression_us": (compressed - start) / 1000,
        "write_sync_us": (finished - compressed) / 1000,
        "write_latency_us": latency_us,
        "cpu_time_us": cpu_us,
        "storage_savings_fraction": 1 - len(saved) / len(data),
        "throughput_mib_s": len(data) / 2**20 / (latency_us / 1e6),
        "verified": True,
    }

def run(args):
    source = args.input.resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("Input must be a nonempty regular file")
    # A fresh directory prevents overwriting previous results.
    args.output.mkdir(parents=True, exist_ok=False)
    scratch = args.output / "trial_files"
    scratch.mkdir()
    compressor = zstandard.ZstdCompressor(level=3, threads=0)
    decompressor = zstandard.ZstdDecompressor()
    rng = random.Random(args.seed)
    metadata = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(),
        "packages": {name: version(name) for name in ("lz4", "zstandard")},
        "input": str(source), "input_bytes": source.stat().st_size,
        "chunk_bytes": args.chunk_bytes, "repeats": args.repeats,
        "seed": args.seed, "lz4_level": 0, "zstd_level": 3,
        "zstd_worker_threads": 0, "cpu_controls": "not collected",
        "scope": "Development baseline; not final experiments or ML training data",
        "timing": "Compression plus file open/write/flush/fsync/close",
        "excluded": "Input reads, entropy, verification, cleanup, ML inference",
        "storage_metric": "Encoded file length, not allocated disk blocks",
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    totals = {c: {"input": 0, "stored": 0, "us": 0, "count": 0} for c in CODECS}
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
                    order = list(CODECS)
                    rng.shuffle(order)
                    for position, codec in enumerate(order):
                        destination = scratch / f"r{repeat}_c{chunk_index}_{codec}.bin"
                        metrics = measure(data, codec, destination, compressor, decompressor)
                        writer.writerow({
                            "workload_id": args.workload_id or source.name,
                            "decision_id": f"r{repeat}_c{chunk_index}",
                            "repeat": repeat, "chunk_index": chunk_index,
                            "trial_order": position, "codec": codec,
                            "chunk_bytes": len(data), "sample_entropy": entropy,
                            **metrics,
                        })
                        total = totals[codec]
                        total["input"] += len(data)
                        total["stored"] += metrics["stored_bytes"]
                        total["us"] += metrics["write_latency_us"]
                        total["count"] += 1
                        # Keep disk use bounded; failed verification leaves evidence.
                        destination.unlink()
                    chunk_index += 1
            csv_file.flush()
    scratch.rmdir()
    metadata["status"] = "complete"
    metadata["trials"] = sum(t["count"] for t in totals.values())
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    for codec, t in totals.items():
        print(f"{codec:4} | {t['count']} trials | "
              f"savings {1 - t['stored'] / t['input']:.1%} | "
              f"mean latency {t['us'] / t['count']:.1f} us | "
              f"throughput {t['input'] / 2**20 / (t['us'] / 1e6):.2f} MiB/s")
    print(f"All outputs verified. Results: {args.output / 'baseline.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--chunk-bytes", type=positive_int, default=262144)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workload-id")
    parser.add_argument("--output", type=Path, default=Path("artifacts") /
                        datetime.now().strftime("baseline_%Y%m%d_%H%M%S_%f"))
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")