"""Train and test the compression selector."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import statistics
import struct
import time
from pathlib import Path

CODECS = ("none", "lz4", "zstd")
BASIC = ("chunk_bytes", "sample_entropy", "cpu_utilization")
CPU = ("quota_fraction", "period_us", "recent_throttle_fraction")
FEATURES = BASIC + CPU
PILOT_FEATURES = ("chunk_bytes", "sample_entropy", "quota_fraction", "period_us")
TARGETS = ("stored_bytes", "write_latency_us")


def load_data(path, feature_set="auto"):
    """Load three codecs per decision."""
    groups = {}
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        if feature_set not in ("auto", "pilot", "full"):
            raise ValueError("Feature set must be auto, pilot, or full")
        optional = set(FEATURES) - set(PILOT_FEATURES)
        if feature_set == "auto" and columns & optional and not optional <= columns:
            raise ValueError("Missing CPU inputs; use --feature-set pilot")
        use_full = feature_set == "full"
        if feature_set == "auto":
            use_full = set(FEATURES) <= columns
        features = FEATURES if use_full else PILOT_FEATURES
        required = {"workload_id", "decision_id", "codec", *features, *TARGETS}
        missing = required - columns
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        for line, raw in enumerate(reader, 2):
            row = {k: raw[k] for k in ("workload_id", "decision_id", "codec")}
            if not all(row.values()) or row["codec"] not in CODECS:
                raise ValueError(f"Line {line}: invalid identifiers or codec")
            for key in (*features, *TARGETS):
                value = float(raw[key])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Line {line}: invalid {key}")
                row[key] = value
            if row["chunk_bytes"] <= 0 or row["write_latency_us"] <= 0 or row["period_us"] <= 0:
                raise ValueError(f"Line {line}: chunk, latency and period must be positive")
            if row["sample_entropy"] > 8 or any(row[k] > 1 for k in optional if k in row):
                raise ValueError(f"Line {line}: feature outside defined range")
            key = (row["workload_id"], row["decision_id"])
            group = groups.setdefault(key, {})
            if row["codec"] in group:
                raise ValueError(f"Line {line}: duplicate codec for {key}")
            if group:
                reference = next(iter(group.values()))
                if any(row[f] != reference[f] for f in features):
                    raise ValueError(f"Line {line}: mismatched inputs for {key}")
            group[row["codec"]] = row
    if not groups or any(set(g) != set(CODECS) for g in groups.values()):
        raise ValueError("Each decision needs none, lz4, and zstd")
    return groups


def train(groups, features, max_depth=5, min_samples_leaf=2):
    from sklearn.tree import DecisionTreeRegressor

    models = {}
    for codec in CODECS:
        rows = [group[codec] for group in groups]
        inputs = [[row[feature] for feature in features] for row in rows]
        models[codec] = {}
        for target in TARGETS:
            model = DecisionTreeRegressor(
                max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42,
            )
            model.fit(inputs, [row[target] for row in rows])
            tree = model.tree_
            models[codec][target] = {
                "left": tree.children_left.tolist(),
                "right": tree.children_right.tolist(),
                "feature": tree.feature.tolist(),
                "threshold": tree.threshold.tolist(),
                "value": tree.value[:, 0, 0].tolist(),
            }
    return {
        "version": 1, "features": list(features), "models": models,
        "settings": {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "seed": 42},
    }


def predict(model, context):
    # Match sklearn's float32 inputs.
    inputs = []
    for feature in model["features"]:
        value = float(context[feature])
        value = struct.unpack("f", struct.pack("f", value))[0]
        if not math.isfinite(value):
            raise ValueError(f"Invalid input: {feature}")
        inputs.append(value)
    result = {}
    for codec, targets in model["models"].items():
        result[codec] = {}
        for target, tree in targets.items():
            node = 0
            while tree["left"][node] != -1:
                feature = tree["feature"][node]
                if inputs[feature] <= tree["threshold"][node]:
                    node = tree["left"][node]
                else:
                    node = tree["right"][node]
            result[codec][target] = tree["value"][node]
    return result


def load_model(path):
    """Load the JSON model."""
    model = json.loads(Path(path).read_text())
    if model.get("version") != 1 or set(model.get("models", {})) != set(CODECS):
        raise ValueError("Unsupported selector model")
    if tuple(model.get("features", ())) not in (PILOT_FEATURES, FEATURES):
        raise ValueError("Unsupported selector features")
    return model


def split_workloads(groups, holdout_workload=None):
    """Hold out one workload, including all quotas and repeats."""
    workloads = sorted({key[0] for key in groups})
    if len(workloads) < 2:
        raise ValueError("At least two workloads required")
    holdout = holdout_workload or random.Random(42).choice(workloads)
    if holdout not in workloads:
        raise ValueError(f"Unknown held-out workload: {holdout}")
    training = [g for key, g in groups.items() if key[0] != holdout]
    testing = [g for key, g in groups.items() if key[0] == holdout]
    return training, testing, holdout


def choose(predictions, budget_us):
    if not math.isfinite(budget_us) or budget_us <= 0:
        raise ValueError("Latency budget must be positive and finite")
    within_budget = []
    for codec in CODECS:
        if predictions[codec]["write_latency_us"] <= budget_us:
            within_budget.append(codec)

    # Smallest within budget; otherwise fastest.
    if within_budget:
        return min(within_budget, key=lambda codec: (
            predictions[codec]["stored_bytes"],
            predictions[codec]["write_latency_us"], codec,
        ))
    return min(CODECS, key=lambda codec: (
        predictions[codec]["write_latency_us"],
        predictions[codec]["stored_bytes"], codec,
    ))


def evaluate(groups, policies, budget):
    results = {}
    for name, model in policies.items():
        selected = []
        overhead = []
        for group in groups:
            if model is None:
                codec = name
            else:
                start = time.perf_counter_ns()
                codec = choose(predict(model, group["none"]), budget)
                overhead.append((time.perf_counter_ns() - start) / 1000)
            selected.append(group[codec])
        count = len(selected)
        original = sum(row["chunk_bytes"] for row in selected)
        stored = sum(row["stored_bytes"] for row in selected)
        latencies = sorted(row["write_latency_us"] for row in selected)
        misses = sum(latency > budget for latency in latencies)
        counts = {codec: 0 for codec in CODECS}
        for row in selected:
            counts[row["codec"]] += 1
        results[name] = {
            "decisions": count,
            "storage_savings_fraction": 1 - stored / original,
            "mean_write_latency_us": statistics.mean(latencies),
            "p95_write_latency_us": latencies[math.ceil(.95 * count) - 1],
            "budget_miss_fraction": misses / count,
            "replayed_serial_throughput_mib_s": original / sum(latencies) * 1e6 / 2**20,
            "mean_prediction_us": statistics.mean(overhead) if overhead else 0,
            "codec_counts": counts,
        }
    return results


def run(args):
    if not math.isfinite(args.budget_us) or args.budget_us <= 0:
        raise ValueError("Latency budget must be positive and finite")
    log_path = getattr(args, "log", None)
    if log_path is not None and (args.output / "model.json").exists():
        raise ValueError("Logged runs need a new output directory")
    groups = load_data(args.data, args.feature_set)
    workloads = sorted({key[0] for key in groups})
    test_data = getattr(args, "test_data", None)
    if test_data is None:
        training, testing, holdout = split_workloads(groups, args.holdout_workload)
        test_workloads = [holdout]
        training_workloads = [w for w in workloads if w != holdout]
    else:
        if args.holdout_workload:
            raise ValueError("Use --holdout-workload or --test-data")
        test_groups = load_data(test_data, args.feature_set)
        test_workloads = sorted({key[0] for key in test_groups})
        if set(workloads) & set(test_workloads):
            raise ValueError("Training and test workloads overlap")
        training = list(groups.values())
        testing = list(test_groups.values())
        training_workloads = workloads
    first = training[0]["none"]
    features = FEATURES if set(FEATURES) <= first.keys() else PILOT_FEATURES
    basic = tuple(f for f in BASIC if f in features)
    for group in testing:
        if not set(features) <= group["none"].keys():
            raise ValueError("Test data: missing model inputs")
    settings = {
        "max_depth": getattr(args, "max_depth", 5),
        "min_samples_leaf": getattr(args, "min_samples_leaf", 2),
    }
    if any(value < 1 for value in settings.values()):
        raise ValueError("Tree settings must be positive")
    model = train(training, features, **settings)
    data_only_model = train(training, basic, **settings)
    ranges = {}
    for feature in features:
        values = [group["none"][feature] for group in training]
        ranges[feature] = [min(values), max(values)]
    training_info = {
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "training_workloads": training_workloads,
        "test_workloads": test_workloads,
        "training_decisions": len(training),
        "test_decisions": len(testing),
        "seed": 42,
        "training_ranges": ranges,
    }
    model["training"] = training_info
    policies = {codec: None for codec in CODECS}
    policies["ml_selector"] = model
    policies["ml_without_quota_throttling"] = data_only_model
    metrics = evaluate(testing, policies, args.budget_us)
    predictions = [predict(model, group["none"]) for group in testing]
    errors = {}
    for codec in CODECS:
        errors[codec] = {}
        for target in TARGETS:
            differences = []
            for prediction, group in zip(predictions, testing):
                differences.append(abs(prediction[codec][target] - group[codec][target]))
            errors[codec][target] = statistics.mean(differences)
    report = {
        "input_data": str(args.data),
        "budget_us": args.budget_us,
        "tree_settings": settings,
        "evaluation_role": getattr(args, "role", "test"),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **training_info,
        "features": list(features),
        "measurement_rows": len(groups) * len(CODECS),
        "metrics": metrics,
        "prediction_mae": errors,
        "limitations": "Offline replay; prediction timing only. Live overhead excluded.",
    }
    if test_data is not None:
        report["test_data"] = str(test_data)
        report["test_data_sha256"] = hashlib.sha256(test_data.read_bytes()).hexdigest()
        report["test_measurement_rows"] = len(testing) * len(CODECS)
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {"model.json": model, "ablation_model.json": data_only_model,
               "evaluation.json": report}
    for filename, data in outputs.items():
        (args.output / filename).write_text(json.dumps(data, indent=2) + "\n")
    if log_path is not None:
        from compression_ml.experiments import record_run
        record_run(report, args.output, log_path)
    print(f"Training: {', '.join(training_info['training_workloads'])}")
    print(f"Test: {', '.join(test_workloads)}")
    print(f"Decisions: {len(training)} train, {len(testing)} test")
    print(f"Model: {args.output / 'model.json'}")
    print(f"Results: {args.output / 'evaluation.json'}")
    if log_path is not None:
        print(f"Log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--budget-us", type=float, required=True)
    parser.add_argument("--feature-set", choices=("auto", "pilot", "full"), default="auto")
    parser.add_argument("--holdout-workload", help="Test workload (all quotas and repeats)")
    parser.add_argument("--test-data", type=Path, help="Separate final test CSV")
    parser.add_argument("--role", choices=("validation", "test"), default="test")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--log", type=Path, default=Path("experiments/runs.csv"))
    timestamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
    parser.add_argument("--output", type=Path, default=Path("artifacts") / timestamp)
    args = parser.parse_args()
    if not math.isfinite(args.budget_us) or args.budget_us <= 0:
        parser.error("--budget-us must be positive and finite")
    run(args)
