"""Repeat the 128-write integration check."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

from benchmarks.collect_pilot import make_input


def main():
    root = Path.cwd()
    timestamp = datetime.now(timezone.utc).strftime("live_%Y%m%d_%H%M%S_%f")
    output = root / "artifacts" / timestamp
    output.mkdir(parents=True, exist_ok=False)
    source = output / "mixed.bin"
    make_input(source, 2048, random.Random(42))
    reference = json.loads((root / "experiments/local_live_01.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != reference["test_data_sha256"]:
        raise ValueError("Live input hash mismatch")

    log_path = output / "console.log"
    print(f"Log: {log_path}", flush=True)
    with log_path.open("w") as log:
        for quota in (25, 50):
            for codec in ("none", "lz4", "zstd", "ml"):
                destination = output / f"{codec}_q{quota}"
                command = [
                    "systemd-run", "--user", "--wait", "--pipe", "--collect",
                    "--expand-environment=no", "--working-directory", str(root),
                    "-p", f"CPUQuota={quota}%", "-p", "CPUQuotaPeriodSec=100ms",
                    sys.executable, str(root / "benchmarks/collect_measurements.py"),
                    str(source), "--codec", codec, "--chunk-bytes", "1048576",
                    "--repeats", "2", "--workload-id", "local_mixed",
                    "--require-cgroup", "--output", str(destination),
                ]
                if codec == "ml":
                    command += ["--model", str(root / "models/pilot/model.json"), "--budget-us", "1000"]
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                metadata = json.loads((destination / "metadata.json").read_text())
                limits = metadata["cgroup"]["limits"]
                if metadata["status"] != "complete" or metadata["trials"] != 16:
                    raise ValueError("Incomplete live run")
                if limits["quota_fraction"] != quota / 100 or limits["period_us"] != 100000:
                    raise ValueError("Live CPU settings mismatch")
                print(f"{codec}, quota {quota}%: passed", flush=True)
    print("Readback: 128 writes passed")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
