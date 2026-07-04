import csv
import json
import os
import tempfile
from pathlib import Path


def test_scaling_benchmark_repro_artifacts():
    from mlxq.bench import run_scaling_benchmark

    keys = ["MLXQ_BACKEND", "MLXQ_SAVE_PLOTS", "MLXQ_BENCH_WARMUPS", "MLXQ_BENCH_REPEATS", "MLXQ_MEMRAY"]
    old_env = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["MLXQ_BACKEND"] = "sv"
        os.environ["MLXQ_SAVE_PLOTS"] = "0"
        os.environ["MLXQ_BENCH_WARMUPS"] = "1"
        os.environ["MLXQ_BENCH_REPEATS"] = "2"
        os.environ.pop("MLXQ_MEMRAY", None)

        with tempfile.TemporaryDirectory(prefix="mlxq_repro_test_") as tmp:
            out_dir = Path(tmp) / "bench_repro"
            result = run_scaling_benchmark("ghz", [2], simulate_cap=2, out_prefix=str(out_dir))
            assert result["results"][0]["timing"]["measured_repeats"] == 2
            assert result["results"][0]["timing"]["warmups"] == 1

            manifest_path = out_dir / "run_manifest.json"
            raw_json_path = out_dir / "ghz_raw_runs.json"
            raw_csv_path = out_dir / "ghz_raw_runs.csv"
            timing_csv_path = out_dir / "ghz_timing_summary.csv"
            main_json_path = out_dir / "ghz_mlx_quantum.json"

            for path in (manifest_path, raw_json_path, raw_csv_path, timing_csv_path, main_json_path):
                assert path.exists(), f"missing benchmark artifact: {path}"

            manifest = json.loads(manifest_path.read_text())
            assert manifest["benchmark_protocol"]["summary_statistic"] == "mean wall-clock milliseconds over measured repeats"
            assert "external_baseline_availability" in manifest

            raw = json.loads(raw_json_path.read_text())
            assert len(raw["raw_runs"]) == 3
            assert sum(1 for item in raw["raw_runs"] if item["warmup"]) == 1
            assert sum(1 for item in raw["raw_runs"] if not item["warmup"]) == 2

            with open(timing_csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert rows[0]["warmups"] == "1"
            assert rows[0]["repeats"] == "2"
            assert float(rows[0]["mean_ms"]) > 0.0

            main = json.loads(main_json_path.read_text())
            assert main["manifest_file"] == "run_manifest.json"
            assert main["results"][0]["timing"]["wall_ms"]["mean"] > 0.0
    finally:
        for key, val in old_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
