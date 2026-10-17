# IBM Z Datathon 2026 — FMR submission draft

**Status:** working draft, based on the supplied participant guidebook and the team-provided email wording, “existing codes and ideas may be used.” The guidebook also has a conflicting prior-work/originality rule (pages 4 and 19–20). Keep the full email as submission evidence and get organizer clarification if the eligibility review requires it.

## Proposed entry

- **Theme:** Theme 1 — Real-Time AI for Critical Decisions
- **Project title:** `[TEAM NAME] | FMR: A Trust Gate for Medical AI`
- **One-line pitch:** FMR checks whether a medical AI answer is supported by its image, then flags uncertain outputs for clinician review.
- **Intended decision:** which AI outputs need priority verification. FMR does not estimate patient acuity, diagnose, or replace clinical judgment.

## Draft form fields

Each draft is under the guidebook’s 1,024-character field limit. Replace `[TEAM NAME]` in the project title before submission.

### Problem statement

Medical vision-language models can produce fluent answers that are weakly grounded in the image. In time-sensitive care, clinicians need a clear signal about which AI outputs should be checked first. FMR checks an answer using image counterfactual sensitivity, reasoning consistency, and a selective-risk gate. It can abstain and route uncertain or uncertified outputs to priority human review. It supports clinical review; it does not assign patient acuity, make a diagnosis, or replace a clinician.

### Project overview / solution

FMR is a model-agnostic verification layer around a medical vision-language model. It tests whether an answer changes when image evidence is removed or altered, measures consistency across sampled reasoning, combines those signals into a faithfulness score, and applies a calibrated answer/abstain gate. The dashboard displays the score, component signals, calibration status, and review route. The current LinuxONE demonstration runs the pipeline on synthetic mock cases; the existing medical-model benchmark outputs come from prior project runs and must be labeled separately. This is a research prototype, not a clinically validated system.

### How the project uses IBM Z / LinuxONE and its value

We adapted FMR’s CPU path for IBM Z (`s390x`) and ran its pipeline in IBM’s LinuxONE machine-learning container. A 4,000-case synthetic run completed in 5.711 seconds on the supplied RHEL 8.10 VM and saved results plus a machine-readable run manifest. MedVLM-R1’s BF16 weights loaded in 102.69 seconds, but one reduced wrapper call did not finish after 5 minutes 25 seconds. A separate SmolVLM-256M general-purpose probe did answer a synthetic shapes question, but its 8-token response took 119 seconds and the model is not medical-trained. Neither run demonstrates medically valid inference or real-time performance on LinuxONE. For a real-model demo, the practical route is to run the existing MedVLM-R1/FMR GPU workflow on Colab and host the dashboard/application side on LinuxONE, clearly label the split, and use only public or de-identified images. If event rules require all inference to stay on LinuxONE, reformulate the model task as finite-choice medical image classification and revalidate the FMR signals; that is a different, not-yet-validated model path.

## Demo video outline (maximum 3 minutes)

1. **0:00–0:25 — Need:** explain that fluent medical AI answers can be visually unsupported and that FMR prioritizes review of AI outputs, not patients.
2. **0:25–0:55 — Method:** show the flow: medical VLM answer → image sensitivity / consistency signals → faithfulness score → answer or abstain → clinician review route.
3. **0:55–1:35 — Prototype:** show the dashboard and one case. Label mock output as synthetic. If showing prior real-model benchmark charts, identify the model, dataset, and that those results were generated before this LinuxONE run.
4. **1:35–2:15 — LinuxONE evidence:** show the `s390x` environment, completed 4,000-case manifest, and measured small-model probe. State that the small model is general-purpose and took 119 seconds for 8 tokens.
5. **2:15–2:45 — Value and limits:** explain local verification as the intended deployment pattern, then state plainly that real-model inference on this VM has not been validated and FMR is not a clinical tool.
6. **2:45–3:00 — Close:** summarize the prototype and show the source repository / project link.

## Submission checklist

- [ ] Complete both required learning labs: Predictive AI / Hybrid AI scenarios (multi-model) and Predictive AI + Generative AI. Save completion evidence.
- [ ] Put the code in an accessible public GitHub/GitLab repository or prepare the permitted ZIP. Include setup instructions, the LinuxONE run command, and the synthetic-data caveat.
- [ ] Add the final project title with the team name and verify all required Hack2Skill text fields stay within 1,024 characters.
- [ ] Prepare the pitch PDF (maximum 5 MB), and an optional architecture flowchart or prototype link if useful.
- [ ] Record and upload the demo video (maximum 3 minutes); do not present synthetic output as clinical or real-model performance.
- [ ] Include links and attribution for open-source code, models, and datasets. Keep the organizer email wording and the guidebook eligibility conflict available for review.
- [ ] Capture LinuxONE evidence: `uname -m`, OS/runtime, run command, completed manifest, and dashboard view.


## Existing real-model result status

The committed outputs under `fmr/outputs/real/` are earlier GPU/Colab runs, separate from the LinuxONE synthetic run. Completed baseline files report VQA-RAD accuracy on n=114 (MedVLM-R1 47.4%, Qwen2.5-VL-3B 64.0%) and PathVQA accuracy on n=250 (MedVLM-R1 28.8%, Qwen2.5-VL-3B 27.2%). The corresponding saved FMR calibration files show an infeasible threshold and zero retained test coverage at the selected error targets; they do not demonstrate a useful calibrated operating point. SLAKE is incomplete because its source image files were missing. Do not present these results as LinuxONE measurements or as clinical validation.

## Verified LinuxONE run

- Platform: Linux on IBM Z, `s390x`; RHEL 8.10 VM.
- Container: `registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest`.
- Runtime: Python 3.12.15; CPU PyTorch image.
- Workload: synthetic `MockVLM`, 4,000 samples, 5 consistency samples; elapsed 5.711 seconds.
- Result: `outputs/linuxone_demo/linuxone_run_manifest.json`, `fmr_results.json`, and `fmr_records.json`.
- This verifies the FMR pipeline and artifact generation on LinuxONE. It does not verify medical-model inference, clinical accuracy, or a real-time clinical workflow.

## Immediate next work

1. Finish and save evidence for the two required labs.
2. Reproduce the LinuxONE run from a clean checkout and capture the short demo.
3. Run the existing real-model workflow in `notebooks/colab_real_pipeline.ipynb` on a GPU if the event permits a split deployment; keep the LinuxONE dashboard and application components clearly identified.
4. If the rules require local LinuxONE inference, prototype a finite-choice medical image/text classifier as a separate model path and revalidate it on medical data before presenting any clinical claim.
5. Publish the source and submission materials, then fill in team details and final links.

## References

- IBM LinuxONE ML environment: <https://github.com/linuxone-community-cloud/jupyter-lab-ml>
- Existing model card and weight/runtime details: <https://huggingface.co/JZPeterPan/MedVLM-R1>
- Small-model LinuxONE probe: <https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct>
- Participant guidebook: `Participant Guidebook_IBM Z Datathon 2026.pdf` in the provided folder.
