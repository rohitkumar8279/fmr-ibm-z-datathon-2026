# FMR on LinuxONE: Datathon Run Notes

## Theme 1 framing

Present FMR as a clinical AI **review-priority aid**: it checks whether a medical
VLM answer is supported by the image and sends low-faithfulness or uncalibrated
answers to a clinician for review. It does not estimate patient acuity, diagnose,
or replace clinical judgment. The current app accepts one image and one question;
it does not yet provide a streaming hospital case queue.

## Runtime facts for the attached VM

- Architecture: IBM Z / `s390x`.
- OS: Red Hat Enterprise Linux 8.10.
- Available memory: about 7.3 GiB.
- The system Python and the existing `~/fmr-env` are Python 3.6.8, while FMR
  requires Python 3.10 or newer. That environment cannot run this project.
- The official LinuxONE ML image provides a newer Jupyter/Python environment and
  CPU PyTorch. The official RHEL instructions use Podman and the s390x Jupyter
  image; the current VM is RHEL 8.10, while the guide's named image is RHEL 9.8.

## First LinuxONE execution

Use the official LinuxONE s390x ML image. The runner can be launched directly in
the container; JupyterLab is not required for this CPU smoke demonstration:

```bash
podman run --rm --user 0:0 \
  -v "$PWD:/workspace" -w /workspace \
  --entrypoint python \
  registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest \
  scripts/run_linuxone_demo.py --samples 4000 --consistency-samples 5
```

The default run uses 4,000 synthetic cases and the offline `MockVLM`. It writes
`outputs/linuxone_demo/fmr_results.json`, `fmr_records.json`, and
`linuxone_run_manifest.json`. The manifest records the architecture and runtime
so a demo can show that this pipeline ran on LinuxONE.

This is a **platform smoke demonstration**, not a real medical-model result.
The project README's real VLM path uses a GPU/Colab workflow. The VM has roughly
7.3 GiB RAM and no accelerator, so MedVLM-R1/Qwen inference on this host is not
assumed to fit or meet a real-time latency target. Measure it before claiming
live real-model performance.

The live API now selects CPU/BF16 defaults on `s390x`, includes the LinuxONE
architecture in `/health`, and returns a `review_routing` result. For a private
API demo, bind the service to the VM loopback and use an SSH tunnel; do not open
the API port publicly. A real-model request may still exceed the VM's memory or
latency budget.

## Latest real-model attempt

The PIL image-processor preflight succeeded in 6.67 seconds without
`torchvision`. MedVLM-R1 then loaded on CPU in BF16 in 102.69 seconds. At that
point the Python process used about 4.98 GB RSS and the host had about 1.5 GiB
available. A single wrapper call using only a 256-pixel generated gray image
and a 64-token limit was still running after 5 minutes 25 seconds, so it was
interrupted. It produced no answer, FMR score, or completed inference latency.
The full multi-pass request was not run. Treat this as a model-load and
preprocessing result only, not a successful real-model inference. The 4.2 GB
model cache remains outside the project in `/home/linux1/shared/fmr-datathon-2026-10-17/hf-cache`.

The attempt record is `outputs/linuxone_demo/linuxone_real_probe_attempt.json`;
the successful processor record is
`outputs/linuxone_demo/linuxone_processor_preflight.json`.

## Smaller-model alternative measured on LinuxONE

I also measured Hugging Face's
[SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct),
a general-purpose image-and-text model (not a medical model). On the same
`s390x` VM, with four CPU threads and a synthetic colored-shapes image:

- FP32 model load: 3.61 seconds; peak process RSS after load: about 2.0 GiB.
- Generating 8 tokens: 119.22 seconds; peak RSS after generation: about 2.37 GiB.
- Output: `The shapes are blue and red.` This checks image-text inference only;
  it does not validate medical reading or FMR.
- A BF16 run capped at 32 tokens was still generating after 4 minutes and was
  stopped; it produced no answer.

The successful run is recorded in
`outputs/linuxone_demo/linuxone_small_vlm_fp32_probe.json`; the incomplete BF16
attempt is recorded in `outputs/linuxone_demo/linuxone_small_vlm_attempts.json`.
Reproduce the short successful probe from the project root on the VM with:

```bash
podman run --rm --user 0:0 \
  -e HF_HOME=/hf-cache \
  -e PYTHONPATH=/hf-cache/pydeps \
  -v /home/linux1/shared/fmr-datathon-2026-10-17/fmr-run/project:/workspace \
  -v /home/linux1/shared/fmr-datathon-2026-10-17/hf-cache:/hf-cache \
  -w /workspace \
  --entrypoint python \
  registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest \
  scripts/probe_linuxone_small_vlm.py
```

Even this 256M model takes about two minutes to emit eight tokens. FMR's current
wrapper adds three answer-scoring forwards per generation, and the standard
request uses eight wrapper calls. This general-purpose model is therefore not a
practical medical replacement on this CPU host.

## Recommended deployment split

Keep MedVLM-R1 as the medical answer model on a GPU host and use LinuxONE for the
application front end, orchestration, audit log, and result dashboard. The repo
already has `notebooks/colab_real_pipeline.ipynb` for real-model benchmark runs,
`api.py` for a live FMR endpoint, and `colab_serve_api.py` as a Colab GPU serving
starter. In this split, the model plus full FMR request runs on the GPU endpoint;
LinuxONE hosts the demo and displays the returned result. Do not claim that the
model or FMR inference ran on LinuxONE.

For a fully on-LinuxONE option, the task would need to change from open-ended
medical reasoning to closed-set image/text similarity with a medical encoder
(for example MedSigLIP or BiomedCLIP) and a finite list of candidate answers.
That would require replacing the rationale/consistency parts of FMR and
validating the new method on medical data; it is not a drop-in substitute for
MedVLM-R1. Google's [MedSigLIP model card](https://huggingface.co/google/medsiglip-448)
recommends it for classification and retrieval without text generation, while
recommending MedGemma for generation. Its checkpoint is gated and listed as
3.51 GB, so I have not claimed or measured it on this VM.
[Microsoft BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
is another medical image-text encoder with a roughly 786 MB checkpoint, but the
model card uses OpenCLIP and OpenCLIP declares `torchvision` as a dependency.
The supplied LinuxONE image lacks `torchvision`, so that path needs a compatible
runtime build before it can be evaluated. Neither encoder produces a reasoning
chain, and neither is a drop-in FMR backend.

## Measure the real MedVLM-R1 path

The probe script separates model load time, one `HFVLM.generate()` call, and
optionally the full FMR score. Each `HFVLM.generate()` does one text generation
and scores three candidate answers, so it makes about four model forwards. A
full request with three counterfactual variants and five consistency samples
therefore makes about 32 model forwards. Run the full request only after the
single-call probe succeeds.

The first run downloads roughly 4.4 GB of weights. Keep that cache outside the
project so it is persistent on the VM and does not get included in a source ZIP.
Transformers also needs `accelerate` for the CPU `device_map` path; install it
without dependencies into the persistent cache overlay. FMR uses the available
Qwen2-VL PIL image processor when `torchvision` is absent. Run these on the VM
after syncing the updated project files:

```bash
mkdir -p /home/linux1/shared/fmr-datathon-2026-10-17/hf-cache
free -h
df -h /home/linux1/shared

podman run --rm --user 0:0 \
  -v /home/linux1/shared/fmr-datathon-2026-10-17/hf-cache:/hf-cache \
  --entrypoint python \
  registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest \
  -m pip install --no-deps --target /hf-cache/pydeps accelerate

podman run --rm --user 0:0 \
  -e HF_HOME=/hf-cache \
  -e PYTHONPATH=/hf-cache/pydeps \
  -v /home/linux1/shared/fmr-datathon-2026-10-17/fmr-run/project:/workspace \
  -v /home/linux1/shared/fmr-datathon-2026-10-17/hf-cache:/hf-cache \
  -w /workspace \
  --entrypoint python \
  registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest \
  scripts/probe_linuxone_real_model.py --dtype bf16 \
  --max-image-side 256 --max-new-tokens 64
```

The default gray image is only for measuring runtime; it cannot validate medical
answers. If load and one call complete with memory headroom, repeat with a
licensed, de-identified public image and run a low-cost full-pipeline check with
one consistency sample (about 16 model forwards):

```bash
podman run --rm --user 0:0 \
  -e HF_HOME=/hf-cache \
  -e PYTHONPATH=/hf-cache/pydeps \
  -v /home/linux1/shared/fmr-datathon-2026-10-17/fmr-run/project:/workspace \
  -v /home/linux1/shared/fmr-datathon-2026-10-17/hf-cache:/hf-cache \
  -w /workspace \
  --entrypoint python \
  registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest \
  scripts/probe_linuxone_real_model.py --dtype bf16 --full-fmr \
  --consistency-samples 1 --max-image-side 256 --max-new-tokens 64 \
  --report outputs/linuxone_fmr_probe_lowcost.json
```

If that succeeds with memory headroom, measure the API-sized request:

```bash
podman run --rm --user 0:0 \
  -e HF_HOME=/hf-cache \
  -e PYTHONPATH=/hf-cache/pydeps \
  -v /home/linux1/shared/fmr-datathon-2026-10-17/fmr-run/project:/workspace \
  -v /home/linux1/shared/fmr-datathon-2026-10-17/hf-cache:/hf-cache \
  -w /workspace \
  --entrypoint python \
  registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest \
  scripts/probe_linuxone_real_model.py --dtype bf16 --full-fmr \
  --consistency-samples 5 --max-image-side 512 --max-new-tokens 256 \
  --report outputs/linuxone_full_fmr_probe.json
```

The first report's load time includes the Hub download; rerun after caching to
measure load from disk. Reports include process high-water RSS, available host
RAM, any visible container memory limit, and inference timings. Rootless Podman
stats is unsupported on this VM's current cgroups setup; monitor from a second
SSH session with `free -h` and `ps -C python -o pid,etime,%cpu,%mem,rss,args`.
Do not retry with FP32 on this 7.3 GiB VM if BF16 fails; FP32 roughly doubles
weight memory and may trigger an out-of-memory kill. The model card's example
uses GPU-oriented Flash Attention; this CPU probe intentionally omits that
option.

## Datathon proof points to capture

1. The LinuxONE VM architecture and resource summary.
2. JupyterLab running on the s390x ML image.
3. The FMR command and completed run manifest.
4. The ANSWER/ABSTAIN result and the honest synthetic-data caveat.
5. The model-load and processor-preflight evidence, clearly labeled incomplete;
   capture a completed inference only if it finishes on the target environment.

## Eligibility note

The organizer email wording quoted by the team says, “existing codes and ideas
may be used.” The participant guidebook also contains a conflicting originality
rule requiring code to be built during the event and disallowing reuse of prior
projects as the core submission. Keep the email with the submission records and
ask the official helpdesk to confirm that it applies to this project if needed.
