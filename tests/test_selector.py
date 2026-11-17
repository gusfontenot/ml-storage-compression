import csv
from argparse import Namespace
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from examples.make_demo_data import generate
from compression_ml.selector import (
    BASIC, CODECS, FEATURES, PILOT_FEATURES,
    choose, evaluate, load_data, load_model, predict, run, split_workloads, train,
)


class SelectorTests(unittest.TestCase):
    def test_budget_and_fallback(self):
        predictions = {c: {"stored_bytes": s, "write_latency_us": t}
                       for c, s, t in (("none", 100, 1), ("lz4", 60, 3), ("zstd", 30, 7))}
        self.assertEqual(choose(predictions, 4), "lz4")
        self.assertEqual(choose(predictions, 10), "zstd")
        self.assertEqual(choose(predictions, .5), "none")
        with self.assertRaises(ValueError):
            choose(predictions, float("nan"))

    def test_pipeline_and_export_parity(self):
        from sklearn.tree import DecisionTreeRegressor
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.csv"
            generate(path)
            data = load_data(path)
        training = [g for (w, _), g in data.items() if w != "synthetic-7"]
        testing = [g for (w, _), g in data.items() if w == "synthetic-7"]
        model = train(training, FEATURES)
        for codec in CODECS:
            rows = [g[codec] for g in training]
            for target in ("stored_bytes", "write_latency_us"):
                reference = DecisionTreeRegressor(max_depth=5, min_samples_leaf=2, random_state=42)
                reference.fit([[r[f] for f in FEATURES] for r in rows], [r[target] for r in rows])
                expected = reference.predict([[g["none"][f] for f in FEATURES] for g in testing])
                for group, value in zip(testing, expected):
                    self.assertAlmostEqual(predict(model, group["none"])[codec][target], value)
        metrics = evaluate(testing, {"none": None, "ml": model, "ablated": train(training, BASIC)}, 1000)
        self.assertEqual(metrics["none"]["storage_savings_fraction"], 0)
        for result in metrics.values():
            self.assertEqual(sum(result["codec_counts"].values()), 30)
            self.assertGreater(result["replayed_serial_throughput_mib_s"], 0)

    def test_reject_invalid_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.csv"
            generate(path)
            with path.open() as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                original = list(reader)
            for mutation in ("missing", "duplicate", "nan", "mismatch"):
                rows = [dict(row) for row in original]
                if mutation == "missing":
                    rows.pop()
                elif mutation == "duplicate":
                    rows.append(dict(rows[0]))
                elif mutation == "nan":
                    rows[0]["write_latency_us"] = "nan"
                else:
                    rows[0]["period_us"] = "1234"
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    load_data(path)

    def test_four_feature_data_and_workload_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.csv"
            generate(path)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            fields = ["workload_id", "decision_id", "codec", *PILOT_FEATURES, "stored_bytes", "write_latency_us"]
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            groups = load_data(path)
            self.assertNotIn("cpu_utilization", next(iter(groups.values()))["none"])
            training, testing, holdout = split_workloads(groups, "synthetic-7")
            self.assertEqual(holdout, "synthetic-7")
            self.assertEqual(len(testing), 30)
            self.assertEqual({g["none"]["workload_id"] for g in testing}, {holdout})
            self.assertNotIn(holdout, {g["none"]["workload_id"] for g in training})
            test_quotas = {group["none"]["quota_fraction"] for group in testing}
            expected_quotas = {
                group["none"]["quota_fraction"]
                for key, group in groups.items() if key[0] == holdout
            }
            self.assertEqual(test_quotas, expected_quotas)
            with self.assertRaises(ValueError):
                load_data(path, "full")
            with self.assertRaises(ValueError):
                split_workloads(groups, "unknown")
            output = Path(directory) / "models"
            with contextlib.redirect_stdout(io.StringIO()):
                run(Namespace(
                    data=path, feature_set="pilot", holdout_workload=holdout,
                    budget_us=1000, output=output,
                ))
            model = load_model(output / "model.json")
            report = json.loads((output / "evaluation.json").read_text())
            self.assertEqual(model["features"], list(PILOT_FEATURES))
            self.assertEqual(model["training"]["test_workloads"], [holdout])
            self.assertEqual(report["test_decisions"], 30)
            self.assertEqual(report["measurement_rows"], 720)
            self.assertEqual(set(predict(model, testing[0]["none"])), set(CODECS))


if __name__ == "__main__":
    unittest.main()
