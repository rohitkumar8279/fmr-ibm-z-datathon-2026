"""Measure a small, general-purpose vision-language model on LinuxONE.

This is a runtime/adapter feasibility probe, not a medical accuracy evaluation.
The synthetic image intentionally contains simple colored shapes rather than
patient data.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path


def rss_peak_mib() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--report", type=Path, default=Path("outputs/linuxone_demo/linuxone_small_vlm_fp32_probe.json"))
    args = parser.parse_args()

    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor

    torch.set_num_threads(args.threads)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    report: dict[str, object] = {
        "model": args.model,
        "dtype": args.dtype,
        "threads": args.threads,
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "medical_model": False,
        "input_kind": "generated colored-shape image; no patient data",
        "max_new_tokens": args.max_new_tokens,
        "scope": "single-image runtime smoke probe; not an FMR run or medical evaluation",
    }
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    report["processor_load_seconds"] = round(time.perf_counter() - started, 3)

    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    load_started = time.perf_counter()
    model = AutoVLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to("cpu").eval()
    report["model_load_seconds"] = round(time.perf_counter() - load_started, 3)
    report["peak_rss_after_load_mib"] = round(rss_peak_mib(), 1)

    image = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 64, 220, 240), fill="red")
    draw.ellipse((286, 230, 440, 384), fill="blue")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe the colored shapes in one short sentence."},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    gen_started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    report["first_generation_seconds"] = round(time.perf_counter() - gen_started, 3)
    report["generated_tokens"] = int(output_ids.shape[-1] - inputs["input_ids"].shape[-1])
    report["answer"] = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )[0].strip()
    report["peak_rss_after_generation_mib"] = round(rss_peak_mib(), 1)
    report["answer_is_medically_validated"] = False
    report["fmr_integration"] = "not tested"

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
