"""Run a reproducible, CPU-only FMR pipeline on IBM Z / LinuxONE.

This uses the project's explicitly synthetic MockVLM fixture. It demonstrates
that the FMR scoring, calibration, and abstention pipeline executes on the
LinuxONE architecture; it is not clinical inference and must not be described
as validation of a real medical VLM.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_fmr
from _common import load_all_configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4000,
                        help="synthetic cases (4000 keeps the project's calibrated demo size)")
    parser.add_argument("--consistency-samples", type=int, default=5,
                        help="repeated mock passes for the consistency signal")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--out", default="outputs/linuxone_demo",
                        help="output directory, relative to this FMR project or absolute")
    args = parser.parse_args()

    if args.samples < 20:
        parser.error("--samples must be at least 20 to keep train/cal/test splits useful")
    if args.consistency_samples < 1:
        parser.error("--consistency-samples must be at least 1")

    configs = load_all_configs()
    data_cfg, model_cfg, exp_cfg = (configs["data"], configs["models"],
                                    configs["experiment"])
    data_cfg["dataset"] = "synthetic"
    data_cfg["synthetic"]["n"] = args.samples
    model_cfg["model"] = "mock_reasoner"
    exp_cfg["signals"]["consistency"]["n_samples"] = args.consistency_samples

    output = Path(args.out)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="fmr_linuxone_cfg_") as temp:
        cfg_dir = Path(temp)
        for name, cfg in (("data", data_cfg), ("models", model_cfg),
                          ("experiment", exp_cfg)):
            (cfg_dir / f"{name}.yaml").write_text(
                yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        started = time.perf_counter()
        result = run_fmr.run(
            str(output), alpha=args.alpha, delta=args.delta,
            post_correction=False, config_dir=str(cfg_dir))
        elapsed = time.perf_counter() - started

    manifest = {
        "run_kind": "synthetic_mock_pipeline",
        "clinical_inference": False,
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "python_version": platform.python_version(),
        "dataset": "synthetic",
        "model": "mock_reasoner",
        "samples": args.samples,
        "consistency_samples": args.consistency_samples,
        "alpha": args.alpha,
        "delta": args.delta,
        "elapsed_seconds": round(elapsed, 3),
        "n_train": result["n_train"],
        "n_cal": result["n_cal"],
        "n_test": result["n_test"],
        "validation": result["validation"],
        "output_files": ["fmr_results.json", "fmr_records.json"],
    }
    (output / "linuxone_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
