"""Generate invented measurements for pipeline checks, never research results."""
import csv
import random
from pathlib import Path
from compression_ml.selector import CODECS, FEATURES, TARGETS


def generate(path):
    rng = random.Random(42)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["workload_id", "decision_id", "codec", *FEATURES, *TARGETS])
        writer.writeheader()
        for workload in range(8):
            for decision in range(30):
                size = rng.choice([65536, 262144, 1048576])
                entropy = rng.uniform(0, 8)
                quota = rng.choice([.25, .5, 1])
                context = dict(zip(FEATURES, (size, entropy, rng.random(), quota, rng.choice([10000, 100000]), rng.random())))
                for codec in CODECS:
                    ratio = 1 if codec == "none" else (.15 if codec == "zstd" else .3) + entropy / 8 * .75
                    latency = size / 1000 * {"none": .2, "lz4": .6, "zstd": 1.8}[codec] / quota
                    writer.writerow({"workload_id": f"synthetic-{workload}", "decision_id": decision,
                                     "codec": codec, **context, "stored_bytes": round(size * ratio),
                                     "write_latency_us": latency * rng.uniform(.9, 1.1)})


if __name__ == "__main__":
    generate(Path("artifacts/synthetic.csv"))
    print("Created artifacts/synthetic.csv — SYNTHETIC, for software checks only.")
