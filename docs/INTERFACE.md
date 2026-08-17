# GI-EndoFM Chat Interface

## Deployment model

GI-EndoFM Chat is a conversational research interface built from three independently
trained components:

1. The SO400M-384 classifier produces the top five HyperKvasir benchmark categories.
2. The seed-43 SigLIP2 model produces a conditional Kvasir-SEG polyp mask.
3. Native Qwen3-VL-8B receives text conversation and optional image context. The
   classification-trained GI LoRA can be enabled as an ablation.

The native Qwen visual tower is retained. The experimental SO400M fixed-64-token
bridge is not loaded because its validation experiment collapsed to one class.

## Optional image context

For each new image, the server:

1. computes a SHA-256 image fingerprint;
2. runs classification and gated segmentation once;
3. stores only the structured evidence in the Gradio session state;
4. sends the image, conversation, and compact evidence to Qwen when an image is
   attached; general text conversation works without an image;
5. reruns the specialist models and clears prior-image history when the attached
   image changes.

There is no application-level keyword refusal router. Uploaded images are written
only to a temporary directory needed by the Qwen image loader and are immediately
removed after that request. Gradio may still maintain its own temporary upload
cache, so identifiable patient information must not be uploaded.

## RunPod installation

Use the validated Python 3.11 / CUDA 12.4 environment:

```bash
cd /workspace/GI-Endoscopy-Grounded-VLM
source /workspace/venvs/qwen3vl/bin/activate

export HF_HOME=/workspace/hf-cache
export PIP_CACHE_DIR=/workspace/pip-cache

python -m pip install -r requirements.txt
python scripts/check_environment.py
python scripts/download_checkpoints.py
```

If the checkpoint repositories are private, authenticate first:

```bash
hf auth login
hf auth whoami
```

Run the complete smoke test:

```bash
python scripts/smoke_test_chat.py
```

Launch the interface:

```bash
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7860
python app.py
```

Conversational generation is question-focused and uses conservative sampling by
default. These settings can be adjusted without changing code:

```bash
export QWEN_DO_SAMPLE=1
export QWEN_TEMPERATURE=0.4
export QWEN_TOP_P=0.9
export QWEN_REPETITION_PENALTY=1.08
```

The selected adapter was trained for GI benchmark classification rather than broad
question answering. Natural conversation therefore defaults to native Qwen while
retaining the specialist evidence and image input:

```bash
export QWEN_CHAT_USE_ADAPTER=0
python app.py
```

Set `QWEN_CHAT_USE_ADAPTER=1` only for a controlled comparison with the
classification-tuned adapter. Text-only conversation works without an attached
image. When an image is attached, specialist outputs are optional context rather
than a mandatory answer template.

The RunPod proxy URL is:

```bash
echo "https://$RUNPOD_POD_ID-7860.proxy.runpod.net"
```

## Optional FlashAttention

The default attention implementation is `sdpa`, which avoids a compiled dependency.
When FlashAttention 2 is already installed and verified, enable it before launch:

```bash
export QWEN_ATTN_IMPLEMENTATION=flash_attention_2
python app.py
```

## Local adapter override

To avoid another Hub download, point directly to the selected checkpoint:

```bash
export QWEN_ADAPTER_PATH=/workspace/qwen-runs/qwen3-vl-8b-lora-sqrt-balanced-seed42/checkpoint-400
python app.py
```

The directory must contain `adapter_config.json` and either
`adapter_model.safetensors` or `adapter_model.bin`.

## Runtime behavior

- Models load lazily on the first image request.
- The Gradio queue permits one GPU request at a time.
- The current image's specialist evidence is reused for later chat turns.
- Attaching a different image invalidates the previous evidence and conversation.
- If Qwen fails and `QWEN_ALLOW_SPECIALIST_FALLBACK=1`, the interface remains
  available using deterministic specialist-evidence responses and reports the
  fallback state in the status line.

## Scientific naming

The UI name is **GI-EndoFM v0.1**. In papers and reports, describe it as a
"GI-specialized multimodal foundation-model research prototype." Do not claim that
it is a clinically validated foundation model or a diagnostic system.
