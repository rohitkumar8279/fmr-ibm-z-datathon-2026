# FMR - Faithful Medical Reasoning

**A review-priority layer for medical vision-language model answers.** FMR checks whether an answer is sensitive to image evidence, whether repeated reasoning stays consistent, and whether the result passes a calibrated answer/abstain gate. Its intended role is to help clinicians decide which AI outputs to verify first. It does not diagnose, estimate patient acuity, or replace clinical judgment.

**IBM Z Datathon 2026 - Theme 1: Real-Time AI for Critical Decisions**  
**Team:** add the registered team name before submitting.

## Current status

The code and dashboard are in this repository. The measured LinuxONE run verifies the synthetic FMR pipeline on IBM Z; the medical-model benchmarks in `fmr/outputs/real/` were run earlier on GPU/Colab and are not LinuxONE results. The current evidence does not establish real-time medical inference on LinuxONE or clinical validity.

| Evidence | Result | What it supports |
|---|---:|---|
| LinuxONE synthetic pipeline | 4,000 MockVLM cases in 5.711 s; `s390x`, Python 3.12.15 | FMR pipeline and artifact generation run on LinuxONE; no medical inference |
| MedVLM-R1 CPU probe on LinuxONE | Weights loaded in 102.69 s; one reduced generation did not finish in 325 s | Model loading/preflight only; no answer or FMR score |
| SmolVLM-256M CPU probe on LinuxONE | 8 tokens in 119.22 s | General image-text runtime only; not a medical model or FMR run |
| Prior GPU/Colab VQA-RAD baseline | n=114; MedVLM-R1 47.4%, Qwen2.5-VL-3B 64.0% | Historical model benchmark, not LinuxONE performance |
| Prior GPU/Colab PathVQA baseline | n=250; MedVLM-R1 28.8%, Qwen2.5-VL-3B 27.2% | Historical model benchmark, not LinuxONE performance |

The saved real-data FMR calibration outputs currently retain **0 test cases** at their configured error targets. The gate therefore abstains on every test case in those runs. SLAKE is incomplete because the source image assets were missing. These files are retained as research evidence and must not be described as validated risk control. Details and run records are in [`fmr/docs/IBM_Z_DATATHON_SUBMISSION_DRAFT.md`](fmr/docs/IBM_Z_DATATHON_SUBMISSION_DRAFT.md) and [`fmr/docs/LINUXONE_DATATHON_RUN.md`](fmr/docs/LINUXONE_DATATHON_RUN.md).

## Run the LinuxONE demonstration

From the repository root on the LinuxONE VM, run the project in the provided s390x container:

```bash
cd fmr
podman run --rm --user 0:0 \
  -v "$PWD:/workspace" -w /workspace \
  --entrypoint python \
  registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest \
  scripts/run_linuxone_demo.py --samples 4000 --consistency-samples 5
```

It writes `fmr/outputs/linuxone_demo/fmr_results.json`, `fmr_records.json`, and `linuxone_run_manifest.json`. The data are generated synthetic cases and `MockVLM`; this run is not medical inference. The archived run manifest and LinuxONE model probes are already included in the repository.

For a local CPU smoke run with Python 3.10 or newer:

```bash
python -m pip install -e fmr
python fmr/scripts/run_linuxone_demo.py --samples 4000 --consistency-samples 5
```

## View the dashboard

Open [`fmr/dashboard/index.html`](fmr/dashboard/index.html) in a browser. It is a static dashboard with no build step. Use the source selector to distinguish the earlier model benchmark data from the synthetic demo data.

## Optional live GPU API demo

`fmr/colab_serve_api.py` is an experimental Colab/ngrok starter. Its API currently has no user authentication and creates a public tunnel. Use only generated, public, or properly de-identified images; never send patient-identifiable data, and stop the tunnel after the demo.

## Reproduce prior model benchmarks

`fmr/notebooks/colab_real_pipeline.ipynb` contains the GPU/Colab workflow. Use Colab's Secrets panel for `HF_TOKEN` and `GH_TOKEN`; do not paste credentials into notebook source. The notebook reads the public `main` branch, and the results helper pushes artifacts to this repository. The current notebook is a starting point for a new run; the committed `fmr/outputs/real/` files are earlier outputs. The LinuxONE VM measurements above show that this GPU workflow has not been reproduced as medical-model inference on the supplied CPU-only VM.

Other notebooks are retained for the project's research and training workflow. Their dependencies and compute requirements vary.

## Submission files and remaining actions

- Pitch PDF: [`output/pdf/FMR_IBM_Z_Datathon_Pitch.pdf`](output/pdf/FMR_IBM_Z_Datathon_Pitch.pdf) (3 pages, under 5 MB).
- Form text, video outline, evidence limitations, and checklist: [`fmr/docs/IBM_Z_DATATHON_SUBMISSION_DRAFT.md`](fmr/docs/IBM_Z_DATATHON_SUBMISSION_DRAFT.md).
- LinuxONE commands and probe records: [`fmr/docs/LINUXONE_DATATHON_RUN.md`](fmr/docs/LINUXONE_DATATHON_RUN.md).
- Before portal submission: replace the team-name placeholder, complete the two required learning labs and save proof, record/upload the required video of at most 3 minutes, and add the video/source links to the submission form.
- The provided organizer email wording says existing code and ideas may be used; the guidebook also contains a conflicting originality rule. Keep both documents available and request organizer clarification if the eligibility review requires it.

## Models and data sources

- [MedVLM-R1 model card](https://huggingface.co/JZPeterPan/MedVLM-R1)
- [Qwen2.5-VL-3B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [VQA-RAD dataset](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)
- [PathVQA dataset](https://huggingface.co/datasets/flaviagiammarino/path-vqa)
- [LinuxONE ML environment](https://github.com/linuxone-community-cloud/jupyter-lab-ml)

Check each model and dataset's current terms before redistributing weights or data. This repository contains source code and result records, not model weights or dataset images. No separate project license was provided with the supplied files.
