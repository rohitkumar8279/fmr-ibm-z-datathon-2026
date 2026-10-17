"""Measure MedVLM-R1 load and FMR inference on IBM Z/LinuxONE.

The default run loads the real model and times one HFVLM.generate() call. That
wrapper call includes one generation plus three candidate-scoring forwards. Use
--full-fmr only after the single-call run succeeds; the default FMR request
uses eight wrapper calls (about 32 model forwards) at five consistency samples.

Without --image, this uses a generated gray image. That is suitable for a
runtime/memory probe only and says nothing about medical accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _proc_memory() -> dict[str, Any]:
    """Return process RSS/high-water mark, host RAM and container memory stats."""
    values: dict[str, Any] = {}
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        for key in ("VmRSS", "VmHWM"):
            match = re.search(rf"^{key}:\s+(\d+)\s+kB", status, re.MULTILINE)
            if match:
                values[key.lower() + "_mb"] = round(int(match.group(1)) / 1024, 1)
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        match = re.search(r"^MemAvailable:\s+(\d+)\s+kB", meminfo, re.MULTILINE)
        if match:
            values["host_mem_available_mb"] = round(int(match.group(1)) / 1024, 1)
        for name, path in (
            ("container_memory_current_mb", Path("/sys/fs/cgroup/memory.current")),
            ("container_memory_limit", Path("/sys/fs/cgroup/memory.max")),
            ("container_memory_usage_mb", Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")),
            ("container_memory_limit_bytes", Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")),
        ):
            if path.exists():
                raw = path.read_text(encoding="utf-8").strip()
                if raw.isdigit():
                    value = int(raw)
                    values[name] = round(value / (1024 * 1024), 1) if name.endswith("_mb") else value
                else:
                    values[name] = raw
    except OSError as exc:
        values["memory_read_error"] = str(exc)
    return values


def _version(package: str) -> str | None:
    try:
        module = __import__(package)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:  # report preflight issue, then fail at actual load
        return f"unavailable: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("FMR_MODEL_ID", "JZPeterPan/MedVLM-R1"))
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--max-image-side", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--consistency-samples", type=int, default=5)
    parser.add_argument("--full-fmr", action="store_true", help="also run the full FMR score after the single-call probe")
    parser.add_argument("--processor-only", action="store_true", help="check Qwen2-VL PIL preprocessing without loading model weights")
    parser.add_argument("--image", type=Path, help="optional de-identified image; omit for a generated gray runtime-probe image")
    parser.add_argument("--question", default="Is an abnormality visible in this image?")
    parser.add_argument("--report", type=Path, default=_ROOT / "outputs" / "linuxone_real_probe.json")
    args = parser.parse_args()

    from PIL import Image
    from fmr.models.hf_vlm import HFVLM
    from fmr.types import Sample

    if args.image:
        image = Image.open(args.image).convert("RGB")
    else:
        image = Image.new("RGB", (args.max_image_side, args.max_image_side), (127, 127, 127))

    sample = Sample(
        sample_id="linuxone_real_model_probe",
        question=args.question,
        answer="",
        modality="xray",
        image=image,
        answer_choices=["yes", "no", "uncertain"],
    )
    report: dict[str, Any] = {
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": _version("torch"),
        "transformers_version": _version("transformers"),
        "model": args.model,
        "device": "cpu",
        "dtype": args.dtype,
        "max_image_side": args.max_image_side,
        "max_new_tokens": args.max_new_tokens,
        "consistency_samples": args.consistency_samples if args.full_fmr else None,
        "input_kind": "provided_image" if args.image else "generated_gray_runtime_probe_only",
        "hf_home": os.environ.get("HF_HOME"),
        "memory_before_load": _proc_memory(),
    }

    vlm = HFVLM(
        model_id=args.model,
        device="cpu",
        dtype=args.dtype,
        is_reasoning=True,
        max_new_tokens=args.max_new_tokens,
        max_image_side=args.max_image_side,
    )
    if args.processor_only:
        started = time.perf_counter()
        vlm._processor = vlm._load_pil_qwen2vl_processor(
            args.max_image_side * args.max_image_side,
            ImportError("PIL processor preflight requested"),
        )
        prompt = vlm._processor.apply_chat_template(
            vlm._messages(sample, image), tokenize=False, add_generation_prompt=True,
        )
        encoded = vlm._processor(text=[prompt], images=[image], return_tensors="pt")
        report["processor_only_seconds"] = round(time.perf_counter() - started, 2)
        report["processor_only_shapes"] = {
            key: list(value.shape) for key, value in encoded.items() if hasattr(value, "shape")
        }
        report["memory_after_processor_only"] = _proc_memory()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        print(f"Saved report: {args.report}", flush=True)
        return

    print(f"Loading {args.model} on CPU ({args.dtype}); first run downloads the weights.", flush=True)
    started = time.perf_counter()
    vlm._ensure_loaded()  # deliberate: time model/processor load separately from inference
    report["load_seconds"] = round(time.perf_counter() - started, 2)
    report["memory_after_load"] = _proc_memory()
    print(f"Model loaded in {report['load_seconds']} seconds; running one wrapper call.", flush=True)

    started = time.perf_counter()
    output = vlm.generate(sample, variant="original", temperature=0.0)
    report["one_generate_seconds"] = round(time.perf_counter() - started, 2)
    report["one_generate_forward_passes"] = 4  # generation + 3 candidate scores
    report["one_generate_answer"] = output.answer
    report["memory_after_one_generate"] = _proc_memory()

    if args.full_fmr:
        from fmr.faithfulness import DEFAULT_WEIGHTS, compute_faithfulness

        print(
            f"Running full FMR with {args.consistency_samples} consistency samples; "
            f"this is about {4 * (3 + args.consistency_samples)} model forwards.",
            flush=True,
        )
        started = time.perf_counter()
        result = compute_faithfulness(
            vlm,
            sample,
            weights=DEFAULT_WEIGHTS,
            n_consistency_samples=args.consistency_samples,
            consistency_temperature=0.7,
        )
        report["full_fmr_seconds"] = round(time.perf_counter() - started, 2)
        report["full_fmr_forward_passes_estimate"] = 4 * (3 + args.consistency_samples)
        report["full_fmr"] = {
            "answer": result["answer"],
            "faithfulness_score": result["fs"],
            "signal_a": result["signal_a"],
            "signal_b": result["signal_b"],
            "signal_c": result["signal_c"],
        }
        report["memory_after_full_fmr"] = _proc_memory()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
