"""Collection splits and run history."""
from argparse import Namespace
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import collect_pilot
from compression_ml.selector import run
from examples.make_demo_data import generate


class ExperimentTests(unittest.TestCase):
    def test_validation_final_test_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            generate(source)
            with source.open() as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            for name, is_test in (("development", False), ("test", True)):
                with (root / f"{name}.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fields)
                    writer.writeheader()
                    writer.writerows(row for row in rows if (row["workload_id"] == "synthetic-7") == is_test)

            args = Namespace(
                data=root / "development.csv", test_data=None, feature_set="pilot",
                holdout_workload="synthetic-6", role="validation", budget_us=1000,
                output=root / "validation", log=root / "runs.csv",
                max_depth=3, min_samples_leaf=3,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                run(args)
            report = json.loads((args.output / "evaluation.json").read_text())
            self.assertEqual(report["evaluation_role"], "validation")
            self.assertEqual(report["test_workloads"], ["synthetic-6"])
            self.assertNotIn("synthetic-6", report["training_workloads"])
            self.assertNotIn("synthetic-7", report["training_workloads"])
            model_bytes = (args.output / "model.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "new output"):
                run(args)
            self.assertEqual((args.output / "model.json").read_bytes(), model_bytes)

            args.output = root / "final"
            args.role = "test"
            args.test_data = root / "test.csv"
            args.holdout_workload = None
            with contextlib.redirect_stdout(io.StringIO()):
                run(args)
            report = json.loads((args.output / "evaluation.json").read_text())
            self.assertEqual(report["test_workloads"], ["synthetic-7"])
            self.assertEqual(len(report["training_workloads"]), 7)
            self.assertEqual(report["test_decisions"], 30)
            model = json.loads((args.output / "model.json").read_text())
            self.assertEqual(model["settings"]["max_depth"], 3)
            self.assertEqual(model["settings"]["min_samples_leaf"], 3)
            with args.log.open() as stream:
                history = list(csv.DictReader(stream))
            self.assertEqual(len(history), 10)
            self.assertEqual({row["role"] for row in history}, {"validation", "test"})
            self.assertTrue(all(row["model_sha256"] for row in history))

            args.output = root / "invalid"
            args.test_data = args.data
            with self.assertRaisesRegex(ValueError, "overlap"):
                run(args)
            self.assertFalse(args.output.exists())

    def test_collection_exports_reserved_workload(self):
        def fake_trial(benchmark, source, workload, quota, codec, destination):
            return [{
                "workload_id": workload, "decision_id": "r0_c0", "chunk_index": 0,
                "codec": codec, "chunk_bytes": 4096, "sample_entropy": 3,
                "quota_fraction": quota / 100, "period_us": 100000,
                "stored_bytes": 4096, "write_latency_us": 100,
            }]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for expanded in (False, True):
                output = root / str(expanded)
                args = Namespace(expanded=expanded, rounds=2, output=output)
                with (
                    patch.object(collect_pilot, "CHUNKS", 1),
                    patch.object(collect_pilot, "CHUNK_BYTES", 4096),
                    patch.object(collect_pilot, "collect_trial", side_effect=fake_trial),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    collect_pilot.collect(args)
                metadata = json.loads((output / "collection.json").read_text())
                self.assertEqual(metadata["status"], "complete")
                self.assertTrue(metadata["input_sha256"])
                self.assertEqual(metadata["rows"], 108 if expanded else 36)
                if expanded:
                    with (output / "development.csv").open() as stream:
                        development = list(csv.DictReader(stream))
                    with (output / "test.csv").open() as stream:
                        testing = list(csv.DictReader(stream))
                    self.assertEqual(len(development), 96)
                    self.assertEqual(len(testing), 12)
                    self.assertNotIn(collect_pilot.TEST_WORKLOAD, {row["workload_id"] for row in development})
                    self.assertEqual({row["workload_id"] for row in testing}, {collect_pilot.TEST_WORKLOAD})
                    self.assertEqual({float(row["quota_fraction"]) for row in testing}, {.25, .5})
                else:
                    self.assertFalse((output / "test.csv").exists())

    def test_failed_collection_stays_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed"
            args = Namespace(expanded=True, rounds=1, output=output)
            with (
                patch.object(collect_pilot, "CHUNKS", 1),
                patch.object(collect_pilot, "CHUNK_BYTES", 4096),
                patch.object(collect_pilot, "collect_trial", side_effect=ValueError("test")),
            ):
                with self.assertRaises(ValueError):
                    collect_pilot.collect(args)
            metadata = json.loads((output / "collection.json").read_text())
            self.assertEqual(metadata["status"], "incomplete")
            self.assertFalse((output / "test.csv").exists())


if __name__ == "__main__":
    unittest.main()
