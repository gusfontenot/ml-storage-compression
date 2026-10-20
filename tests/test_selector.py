import csv
import tempfile
import unittest
from pathlib import Path
from examples.make_demo_data import generate
from compression_ml.selector import BASIC, CODECS, FEATURES, choose, evaluate, load_data, predict, train


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


if __name__ == "__main__":
    unittest.main()
