"""Check the saved validation results."""
from argparse import Namespace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from compression_ml.selector import load_data, load_model, predict, run


def compare(actual, expected, label):
    for name, value in expected.items():
        if name == "mean_prediction_us":
            continue
        field = f"{label}.{name}"
        if isinstance(value, dict):
            compare(actual[name], value, field)
        elif not math.isclose(actual[name], value, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(f"Result mismatch: {field}")


def main():
    reference = json.loads(Path("models/local_v2_validation/evaluation.json").read_text())
    data = Path(reference["input_data"])
    if hashlib.sha256(data.read_bytes()).hexdigest() != reference["data_sha256"]:
        raise ValueError("Saved CSV hash mismatch")

    timestamp = datetime.now(timezone.utc).strftime("reproduce_%Y%m%d_%H%M%S_%f")
    output = Path("artifacts") / timestamp
    holdout = reference["test_workloads"][0]
    run(Namespace(
        data=data, feature_set="pilot", holdout_workload=holdout,
        budget_us=reference["budget_us"], role="validation",
        max_depth=reference["tree_settings"]["max_depth"],
        min_samples_leaf=reference["tree_settings"]["min_samples_leaf"],
        output=output, log=output / "runs.csv",
    ))
    actual = json.loads((output / "evaluation.json").read_text())
    compare(actual["metrics"], reference["metrics"], "metrics")
    compare(actual["prediction_mae"], reference["prediction_mae"], "prediction_mae")

    old_model = load_model("models/pilot/model.json")
    groups = load_data(data, "pilot")
    validation = [group for (name, _), group in groups.items() if name == holdout]
    for codec in ("lz4", "zstd"):
        errors = []
        for group in validation:
            estimate = predict(old_model, group["none"])[codec]["stored_bytes"]
            errors.append(abs(estimate - group[codec]["stored_bytes"]))
        old_error = statistics.mean(errors)
        new_error = actual["prediction_mae"][codec]["stored_bytes"]
        print(f"{codec} size error: {old_error:,.0f} -> {new_error:,.0f} bytes")
    print("Saved results: matched")
    print("Prediction timing: varies per run")
    print("Final test: unused")


if __name__ == "__main__":
    main()
