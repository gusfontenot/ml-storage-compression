"""Run history."""
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path

FIELDS = (
    "recorded_utc", "run", "role", "data_sha256", "test_data_sha256",
    "model_sha256", "train_workloads", "test_workloads", "budget_us",
    "max_depth", "min_samples_leaf", "policy", "savings_fraction",
    "mean_write_us", "p95_write_us", "budget_miss_fraction",
    "prediction_us", "report",
)


def record_run(report, output, log_path):
    output = Path(output)
    log_path = Path(log_path)
    settings = report.get("tree_settings", {})
    row = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "run": output.name,
        "role": report.get("evaluation_role", "test"),
        "data_sha256": report["data_sha256"],
        "test_data_sha256": report.get("test_data_sha256", report["data_sha256"]),
        "model_sha256": hashlib.sha256((output / "model.json").read_bytes()).hexdigest(),
        "train_workloads": "|".join(report["training_workloads"]),
        "test_workloads": "|".join(report["test_workloads"]),
        "budget_us": report["budget_us"],
        "max_depth": settings.get("max_depth", ""),
        "min_samples_leaf": settings.get("min_samples_leaf", ""),
        "report": str(output / "evaluation.json"),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and log_path.stat().st_size:
        with log_path.open(newline="") as stream:
            if next(csv.reader(stream)) != list(FIELDS):
                raise ValueError("Run log: wrong columns")
    with log_path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        for policy, metrics in report["metrics"].items():
            writer.writerow({
                **row,
                "policy": policy,
                "savings_fraction": metrics["storage_savings_fraction"],
                "mean_write_us": metrics["mean_write_latency_us"],
                "p95_write_us": metrics["p95_write_latency_us"],
                "budget_miss_fraction": metrics["budget_miss_fraction"],
                "prediction_us": metrics["mean_prediction_us"],
            })
