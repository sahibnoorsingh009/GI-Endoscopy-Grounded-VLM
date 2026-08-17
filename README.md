# GI Endoscopy Grounded VLM Interface

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Research use only](https://img.shields.io/badge/use-research%20only-yellow.svg)](#responsible-use)

A reproducible, ChatGPT-style research interface for gastrointestinal endoscopy
images. The deployed model stack combines:

- supervised SigLIP2 SO400M-384 classification over 23 HyperKvasir categories;
- gated SigLIP2 Base NaFlex polyp segmentation;
- native Qwen3-VL-8B for general text and multimodal conversation;
- optional specialist-model evidence when an endoscopy image is attached.

The interface is called **GI-EndoFM v0.1**. It is a GI-specialized multimodal
foundation-model research prototype, not a clinically validated general-purpose
endoscopy foundation model. The unsuccessful fixed-64-token SO400M/Qwen bridge is
preserved only as an ablation and is not used by the interface.

![Interface overview](assets/interface_overview.png)

## Responsible use

This software is a research demonstration. It is not validated for diagnosis,
treatment, patient management, or clinical decision-making. Do not upload
identifiable patient information. Classification softmax scores are not calibrated
clinical probabilities.

## Architecture

    Endoscopy image ---------------------> native Qwen3-VL visual tower
           |                                      |
           v                                      v
    SO400M-384 classifier                 native Qwen3-VL chat
           |                                      |
           v                                      |
    top-five HyperKvasir scores                    |
           |                                      |
           v                                      |
    polyp-family score >= 0.30?                    |
          / \                                     |
        yes  no                                   |
        /     \                                    |
       v       +----> segmentation skipped         |
    seed-43 SigLIP2 segmenter                      |
       |                                           |
       v                                           |
    mask + geometry ----> structured evidence -----+
                                                   |
                                                   v
                                      grounded conversational answer

The two specialist models remain separate because the classification checkpoint is
validated for semantic recognition and the segmentation checkpoint is validated for
pixel localization.

## Checkpoints

Weights are downloaded automatically from Hugging Face on first analysis.
They are never committed to this Git repository.

| Branch | Repository and file | Approximate size |
| --- | --- | ---: |
| Classification | [SO400M seed 42 compact checkpoint](https://huggingface.co/Sahibnoor1/gi-siglip2-dino-hyperkvasir-checkpoints) — checkpoints/siglip2_so400m_384_supervised_v1/seed42/so400m_classifier_seed42_vision_ema.pt | 1.6 GiB |
| Segmentation | [SigLIP2 full seed 43](https://huggingface.co/Sahibnoor1/kvasir-siglip2-segmentation-checkpoints) — checkpoints/siglip2_full/seed_43/best.pt | 388 MB |
| Optional classification LoRA | [GI-EndoFM checkpoints](https://huggingface.co/Sahibnoor1/gi-endoscopy-grounded-vlm-checkpoints) — qwen3-vl-8b-lora/sqrt-balanced-seed42/checkpoint-400 | adapter only |

The optional Qwen adapter uses `Qwen/Qwen3-VL-8B-Instruct` as its base model. The selected
checkpoint achieved validation accuracy 0.4620, macro F1 0.2441, and balanced
accuracy 0.2684 on the fixed HyperKvasir validation split. These benchmark values
do not establish clinical performance. It is disabled for normal conversation by
default because it was trained for classification rather than instruction following.

For a genuinely one-command public demo, both checkpoint repositories must be public.
If they remain private or gated, users need a Hugging Face read token:

    hf auth login

Alternatively, set the HF_TOKEN environment variable. Never commit a token.

## Hardware and software

Tested configuration:

- Ubuntu Linux;
- Python 3.11;
- PyTorch 2.6.0 with CUDA 12.4;
- Transformers 4.57.6;
- NVIDIA RTX 6000 Ada Generation with 48 GiB VRAM.

The complete generative stack requires an NVIDIA GPU. Use at least 24 GiB VRAM;
48 GiB is recommended and is the validated configuration. Approximately 25 GiB of
free disk space is recommended for the Qwen base model, adapters, specialist
checkpoints, dependencies, and caches. If Qwen cannot load, the interface can fall
back to deterministic specialist-evidence answers.

## Local installation

Clone the repository:

    git clone https://github.com/sahibnoorsingh009/GI-Endoscopy-Grounded-VLM.git
    cd GI-Endoscopy-Grounded-VLM

On Linux or macOS, create the environment and install dependencies:

    bash scripts/setup_local.sh
    source .venv/bin/activate

On Windows PowerShell:

    powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
    .\.venv\Scripts\Activate.ps1

Optionally download both checkpoints before starting:

    python scripts/download_checkpoints.py

Run the interface:

    python app.py

Open http://127.0.0.1:7860 in a browser.

Run the end-to-end smoke test:

    python scripts/smoke_test.py

The first model load can take several minutes because the Qwen base model and all
three project checkpoints must be downloaded and initialized. See
[the interface guide](docs/INTERFACE.md) for its runtime and grounding contract.

## RunPod installation

1. Create one GPU Pod using an official PyTorch template.
2. Enable SSH or Jupyter access.
3. In the Pod template, add 7860 to Expose HTTP Ports.
4. Keep the repository and caches under /workspace so they survive normal Pod restarts.

Inside the Pod:

    cd /workspace
    git clone https://github.com/sahibnoorsingh009/GI-Endoscopy-Grounded-VLM.git
    cd GI-Endoscopy-Grounded-VLM
    bash scripts/setup_runpod.sh
    python scripts/download_checkpoints.py
    python app.py

The public RunPod URL follows this pattern:

    https://POD_ID-7860.proxy.runpod.net

You can print it inside the Pod with:

    echo "https://$RUNPOD_POD_ID-7860.proxy.runpod.net"

The HTTP proxy makes the interface public. For basic access control, set both
variables before launching:

    export VLM_USERNAME=researcher
    export VLM_PASSWORD='use-a-long-random-password'
    python app.py

Stop the Pod when it is not needed to stop GPU billing. Back up /workspace before
terminating any Pod or deleting its persistent volume.

## Configuration

| Environment variable | Default | Purpose |
| --- | --- | --- |
| HF_HOME | Hugging Face default | Model cache root |
| HF_TOKEN | unset | Read access for private or gated checkpoints |
| CLASSIFICATION_CHECKPOINT | auto-download | Local classification checkpoint override |
| SEGMENTATION_CHECKPOINT | auto-download | Local segmentation checkpoint override |
| POLYP_GATE_THRESHOLD | 0.30 | Combined polyp-family gate |
| QWEN_MODEL_ID | Qwen/Qwen3-VL-8B-Instruct | Native Qwen base model |
| QWEN_ADAPTER_REPO_ID | Sahibnoor1/gi-endoscopy-grounded-vlm-checkpoints | Adapter model repository |
| QWEN_ADAPTER_SUBFOLDER | qwen3-vl-8b-lora/sqrt-balanced-seed42/checkpoint-400 | Selected LoRA folder |
| QWEN_ADAPTER_PATH | unset | Optional local adapter override |
| QWEN_ATTN_IMPLEMENTATION | sdpa | Use flash_attention_2 only when installed |
| QWEN_MAX_NEW_TOKENS | 256 | Maximum response tokens |
| QWEN_ALLOW_SPECIALIST_FALLBACK | 1 | Deterministic fallback when Qwen cannot load |
| GRADIO_SERVER_NAME | 0.0.0.0 | Interface bind address |
| GRADIO_SERVER_PORT | 7860 | Interface port |
| VLM_USERNAME | unset | Optional basic-auth username |
| VLM_PASSWORD | unset | Optional basic-auth password |

See [.env.example](.env.example) for a complete template.

## Validated segmentation results

On the fixed 120-image official Kvasir-SEG test split, the seed-43 checkpoint achieved:

| Metric | Value |
| --- | ---: |
| Mean Dice | 0.902960 |
| Mean IoU | 0.847710 |
| Precision | 0.922011 |
| Recall | 0.919619 |
| Failure rate (Dice below 0.10) | 0.000000 |

The deployment uses the same 320 by 320 segmentation geometry as training and maps
the predicted logits back to the uploaded image size. See
[validation details](docs/VALIDATION.md) for the spatial-alignment explanation and
the six-image deployment sanity check.

![Classification evidence and grounded Q&A](assets/evidence_qa.png)

## Repository structure

    .
    ├── app.py                     # Gradio entry point
    ├── configs/                   # Segmentation configuration
    ├── demo/examples/images/      # Six research examples
    ├── docs/INTERFACE.md          # Chat architecture and RunPod launch guide
    ├── docs/VALIDATION.md         # Metrics and geometry validation
    ├── scripts/                   # Setup, download, checks, smoke test
    ├── src/models/                # Segmentation architecture
    ├── tests/                     # Lightweight repository checks
    └── vlm_demo/                  # UI and two-branch inference service

## Known limitations

- Classification is limited to the 23 HyperKvasir benchmark categories.
- The segmentation model was trained on Kvasir-SEG polyp images and is not a
  validated no-polyp detector.
- The 0.30 classification gate has not yet been clinically calibrated.
- Mask quality can be poor on difficult or out-of-distribution images.
- Qwen can still produce imperfect language despite its grounding prompt; displayed
  specialist evidence remains the authoritative model output.
- The selected Qwen LoRA has modest 23-class macro F1 and is not a diagnostic model.
- The failed fixed-token distilled bridge is excluded from deployment.
- Performance outside the reported datasets and hardware has not been established.

## Source projects

- [GI-SigLIP2-DINO-HyperKvasir](https://github.com/sahibnoorsingh009/GI-SigLIP2-DINO-HyperKvasir)
- [Kvasir-SigLIP-Segmentation](https://github.com/sahibnoorsingh009/Kvasir-SigLIP-Segmentation)

## License

Code is released under the [Apache License 2.0](LICENSE). Dataset images and
model checkpoints remain subject to their original licenses and terms.
