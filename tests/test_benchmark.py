"""Benchmark tests (simulated cgroup readings)."""
from argparse import Namespace
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import collect_measurements as benchmark
from compression_ml.selector import CODECS, PILOT_FEATURES


def constant_model(selected):
    def leaf(value):
        return {
            "left": [-1], "right": [-1], "feature": [-2],
            "threshold": [-2], "value": [value],
        }
    return {"version": 1, "features": list(PILOT_FEATURES), "models": {
        codec: {"stored_bytes": leaf(1 if codec == selected else 100), "write_latency_us": leaf(10)}
        for codec in CODECS
    }}


def make_monitor(root, quota):
    directory = root / "cgroup"
    directory.mkdir(exist_ok=True)
    (directory / "cpu.max").write_text(f"{quota} 100000\n")
    counters = "".join(f"{key} 0\n" for key in benchmark.STAT_KEYS)
    (directory / "cpu.stat").write_text(counters)
    return benchmark.CgroupCPU(directory)


class BenchmarkTests(unittest.TestCase):
    def test_saved_pilot_model_with_both_quotas(self):
        model_path = Path(__file__).resolve().parents[1] / "models/pilot/model.json"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.bin"
            source.write_bytes(b"ABCDE123" * (1048576 // 8))
            for quota in (25000, 50000):
                monitor = make_monitor(root, quota)
                args = Namespace(input=source, codec="ml", model=model_path, budget_us=1000,
                                 require_cgroup=True, output=root / str(quota), seed=42,
                                 chunk_bytes=1048576, repeats=1, workload_id="test")
                with (
                    patch.object(benchmark, "detect_cgroup", return_value=(monitor, None)),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    benchmark.run(args)
                with (args.output / "baseline.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["verified"], "True")
                self.assertEqual(float(rows[0]["quota_fraction"]), quota / 100000)
                self.assertEqual(rows[0]["outside_training_range"], "False")

    def test_model_routes_to_each_codec_and_verifies_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.bin"
            source.write_bytes(b"hello compression" * 100)
            monitor = make_monitor(root, 25000)
            for selected in CODECS:
                with self.subTest(codec=selected):
                    model_path = root / f"{selected}.json"
                    model_path.write_text(json.dumps(constant_model(selected)))
                    output = root / selected
                    args = Namespace(input=source, codec="ml", model=model_path, budget_us=1000,
                                     require_cgroup=True, output=output, seed=42, chunk_bytes=1000,
                                     repeats=2, workload_id="test")
                    with (
                        patch.object(benchmark, "detect_cgroup", return_value=(monitor, None)),
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        benchmark.run(args)
                    with (output / "baseline.csv").open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(len(rows), 4)
                    self.assertEqual({r["codec"] for r in rows}, {selected})
                    self.assertTrue(all(r["verified"] == "True" for r in rows))
                    self.assertTrue(all(float(r["prediction_us"]) > 0 for r in rows))
                    for row in rows:
                        stages = ("write_latency_us", "feature_collection_us", "prediction_us")
                        measured_time = sum(float(row[field]) for field in stages)
                        self.assertGreaterEqual(float(row["decision_write_latency_us"]), measured_time)
                    metadata = json.loads((output / "metadata.json").read_text())
                    self.assertEqual(metadata["status"], "complete")
                    self.assertEqual(metadata["codec_counts"][selected], 4)
                    self.assertFalse((output / "trial_files").exists())

    def test_baselines_keep_working(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.bin"
            source.write_bytes(b"abc" * 100)
            for codec in ("all", *CODECS):
                args = Namespace(input=source, codec=codec, model=None, budget_us=None,
                                 require_cgroup=False, output=root / codec, seed=42,
                                 chunk_bytes=1000, repeats=1, workload_id="test")
                with (
                    patch.object(benchmark, "detect_cgroup", return_value=(None, "test")),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    benchmark.run(args)
                with (args.output / "baseline.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                expected = set(CODECS) if codec == "all" else {codec}
                self.assertEqual({row["codec"] for row in rows}, expected)
                self.assertTrue(all(float(r["prediction_us"]) == 0 for r in rows))

    def test_ml_rejects_unlimited_quota_and_missing_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.bin"
            source.write_bytes(b"input")
            model_path = root / "model.json"
            model_path.write_text(json.dumps(constant_model("none")))
            args = Namespace(input=source, codec="ml", model=None, budget_us=1000,
                             require_cgroup=False, output=root / "out", seed=42,
                             chunk_bytes=1000, repeats=1, workload_id="test")
            with self.assertRaisesRegex(ValueError, "requires --model"):
                benchmark.run(args)
            args.model = model_path
            monitor = make_monitor(root, "max")
            with patch.object(benchmark, "detect_cgroup", return_value=(monitor, None)):
                with self.assertRaisesRegex(ValueError, "finite CPU quota"):
                    benchmark.run(args)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
