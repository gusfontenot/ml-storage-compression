"""Generate demo data (invented timings)."""
import csv
import random
from pathlib import Path
from compression_ml.selector import CODECS, FEATURES, TARGETS


def generate(path):
    rng = random.Random(42)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        fields = ["workload_id", "decision_id", "codec", *FEATURES, *TARGETS]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for workload in range(8):
            for decision in range(30):
                size = rng.choice([65536, 262144, 1048576])
                entropy = rng.uniform(0, 8)
                quota = rng.choice([.25, .5, 1])
                context = {
                    "chunk_bytes": size,
                    "sample_entropy": entropy,
                    "cpu_utilization": rng.random(),
                    "quota_fraction": quota,
                    "period_us": rng.choice([10000, 100000]),
                    "recent_throttle_fraction": rng.random(),
                }
                for codec in CODECS:
                    if codec == "none":
                        ratio = 1
                    elif codec == "lz4":
                        ratio = .3 + entropy / 8 * .75
                    else:
                        ratio = .15 + entropy / 8 * .75
                    latency = size / 1000 * {"none": .2, "lz4": .6, "zstd": 1.8}[codec] / quota
                    writer.writerow({
                        "workload_id": f"synthetic-{workload}",
                        "decision_id": decision,
                        "codec": codec,
                        **context,
                        "stored_bytes": round(size * ratio),
                        "write_latency_us": latency * rng.uniform(.9, 1.1),
                    })


if __name__ == "__main__":
    generate(Path("artifacts/synthetic.csv"))
    print("Demo CSV: artifacts/synthetic.csv (generated values)")
