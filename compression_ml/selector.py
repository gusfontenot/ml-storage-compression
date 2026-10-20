"""Train and evaluate a compression selector on matched codec measurements."""
import argparse
import csv
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
TARGETS = ("stored_bytes", "write_latency_us")


def load_data(path):
    """Require a complete triplet for each independent pre-decision context."""
    groups = {}
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"workload_id", "decision_id", "codec", *FEATURES, *TARGETS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for line, raw in enumerate(reader, 2):
            row = {k: raw[k] for k in ("workload_id", "decision_id", "codec")}
            if not all(row.values()) or row["codec"] not in CODECS:
                raise ValueError(f"Line {line}: invalid identifiers or codec")
            for key in (*FEATURES, *TARGETS):
                value = float(raw[key])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Line {line}: invalid {key}")
                row[key] = value
            if row["chunk_bytes"] <= 0 or row["write_latency_us"] <= 0 or row["period_us"] <= 0:
                raise ValueError(f"Line {line}: chunk, latency and period must be positive")
            if row["sample_entropy"] > 8 or any(row[k] > 1 for k in ("cpu_utilization", "recent_throttle_fraction")):
                raise ValueError(f"Line {line}: feature outside defined range")
            key = (row["workload_id"], row["decision_id"])
            group = groups.setdefault(key, {})
            if row["codec"] in group:
                raise ValueError(f"Line {line}: duplicate codec for {key}")
            if group and any(row[f] != next(iter(group.values()))[f] for f in FEATURES):
                raise ValueError(f"Line {line}: mismatched pre-decision features for {key}")
            group[row["codec"]] = row
    if not groups or any(set(g) != set(CODECS) for g in groups.values()):
        raise ValueError("Need complete none/lz4/zstd measurements for every decision")
    return groups


def train(groups, features):
    from sklearn.tree import DecisionTreeRegressor

    models = {}
    for codec in CODECS:
        rows = [g[codec] for g in groups]
        x = [[r[f] for f in features] for r in rows]
        models[codec] = {}
        for target in TARGETS:
            model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=2, random_state=42)
            model.fit(x, [r[target] for r in rows])
            tree = model.tree_
            models[codec][target] = {
                "left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
                "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
                "value": tree.value[:, 0, 0].tolist(),
            }
    return {"version": 1, "features": list(features), "models": models}


def predict(model, context):
    # sklearn trees compare float32 inputs; preserve that behavior in the JSON runtime.
    x = [struct.unpack("f", struct.pack("f", float(context[f])))[0] for f in model["features"]]
    if not all(math.isfinite(v) for v in x):
        raise ValueError("Inference features must be finite")
    result = {}
    for codec, targets in model["models"].items():
        result[codec] = {}
        for target, tree in targets.items():
            node = 0
            while tree["left"][node] != -1:
                branch = "left" if x[tree["feature"][node]] <= tree["threshold"][node] else "right"
                node = tree[branch][node]
            result[codec][target] = tree["value"][node]
    return result


def choose(predictions, budget_us):
    if not math.isfinite(budget_us) or budget_us <= 0:
        raise ValueError("Latency budget must be positive and finite")
    feasible = [c for c in CODECS if predictions[c]["write_latency_us"] <= budget_us]
    if feasible:
        return min(feasible, key=lambda c: (predictions[c]["stored_bytes"], predictions[c]["write_latency_us"], c))
    return min(CODECS, key=lambda c: (predictions[c]["write_latency_us"], predictions[c]["stored_bytes"], c))


def evaluate(groups, policies, budget):
    results = {}
    for name, model in policies.items():
        selected, overhead = [], []
        for group in groups:
            if model is None:
                codec = name
            else:
                start = time.perf_counter_ns()
                codec = choose(predict(model, group["none"]), budget)
                overhead.append((time.perf_counter_ns() - start) / 1000)
            selected.append(group[codec])
        original = sum(r["chunk_bytes"] for r in selected)
        latencies = sorted(r["write_latency_us"] for r in selected)
        results[name] = {
            "decisions": len(selected),
            "storage_savings_fraction": 1 - sum(r["stored_bytes"] for r in selected) / original,
            "mean_write_latency_us": statistics.mean(latencies),
            "p95_write_latency_us": latencies[math.ceil(.95 * len(latencies)) - 1],
            "budget_miss_fraction": sum(t > budget for t in latencies) / len(latencies),
            "replayed_serial_throughput_mib_s": original / sum(latencies) * 1e6 / 2**20,
            "mean_prediction_us": statistics.mean(overhead) if overhead else 0,
            "codec_counts": {c: sum(r["codec"] == c for r in selected) for c in CODECS},
        }
    return results


def run(args):
    groups = load_data(args.data)
    workloads = sorted({key[0] for key in groups})
    if len(workloads) < 2:
        raise ValueError("Need at least two workload IDs for a held-out workload split")
    random.Random(42).shuffle(workloads)
    held_out = set(workloads[:max(1, math.ceil(len(workloads) * .25))])
    training = [g for key, g in groups.items() if key[0] not in held_out]
    testing = [g for key, g in groups.items() if key[0] in held_out]
    full, ablated = train(training, FEATURES), train(training, BASIC)
    metrics = evaluate(testing, {**{c: None for c in CODECS}, "ml_full": full, "ml_without_quota_throttling": ablated}, args.budget_us)
    report = {"input_data": str(args.data), "seed": 42, "budget_us": args.budget_us,
              "training_workloads": sorted(set(workloads) - held_out),
              "test_workloads": sorted(held_out), "training_decisions": len(training),
              "metrics": metrics,
              "limitations": "Offline replay; prediction timings are local Python timings. Feature collection, live integration and cgroup overhead are not measured."}
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, data in (("model.json", full), ("ablation_model.json", ablated), ("evaluation.json", report)):
        (args.output / filename).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--budget-us", type=float, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if not math.isfinite(args.budget_us) or args.budget_us <= 0:
        parser.error("--budget-us must be positive and finite")
    run(args)
