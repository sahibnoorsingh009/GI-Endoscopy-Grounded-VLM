# Qwen integration roadmap

The Qwen work is isolated on the `qwen-integration` branch so the validated
classification and segmentation demo remains reproducible.

## Phase 1: native Qwen3-VL baseline

Fine-tune Qwen3-VL-8B-Instruct with LoRA on the official HyperKvasir
classification training split. This is the comparison baseline; it uses Qwen's
native visual tower and does not replace it with the trained SO400M encoder.

The split policy is fixed:

| Split | Images | Policy |
| --- | ---: | --- |
| train | 7,433 | adapter training |
| val | 1,593 | model selection and calibration only |
| test | 1,593 | one-time final evaluation only |

Build the official Qwen annotations and materialize the two required MegaBank
tar shards:

```bash
python scripts/build_qwen_manifest.py \
  --split-csv /workspace/classification-reference/metadata/hyperkvasir_23class_official_70_15_15_split.csv \
  --output-dir /workspace/qwen-data/hyperkvasir \
  --materialize
```

The generated structure is:

```text
/workspace/qwen-data/hyperkvasir/
├── annotations/
│   ├── hyperkvasir_train.json
│   ├── hyperkvasir_val.json
│   └── hyperkvasir_test.json
├── images/
├── index/
│   ├── hyperkvasir_train.csv
│   ├── hyperkvasir_val.csv
│   └── hyperkvasir_test.csv
└── manifest_summary.json
```

The JSON follows the official Qwen VL conversation contract: one `image` path,
one matching `<image>` tag, and alternating `human`/`gpt` turns.

Create the isolated environment and evaluate a balanced 100-image validation
pilot before launching the full validation run:

```bash
python -m venv /workspace/venvs/qwen3vl
source /workspace/venvs/qwen3vl/bin/activate
pip install -r requirements-qwen.txt

python scripts/evaluate_qwen_zero_shot.py \
  --data-root /workspace/qwen-data/hyperkvasir \
  --output-dir /workspace/qwen-results/zero-shot-val-pilot100 \
  --split val \
  --limit 100
```

The test split remains untouched during baseline development and adapter
selection. Evaluation writes each prediction immediately to JSONL and supports
safe continuation with `--resume`.

## Phase 2: trained SO400M to Qwen bridge

The proposed research model will connect frozen SO400M patch tokens to a Qwen3
text decoder through a trainable token resampler and projector. It will be
compared against the Phase 1 native-Qwen vision baseline. The validated seed-43
segmenter initially remains an external expert that supplies masks, regions, and
bounding boxes as grounded evidence.

Phase 2 freezes all three previously selected components: Qwen3-VL-8B,
the balanced native-vision LoRA selected at validation step 400, and the
GI-trained SO400M encoder. Only the 36.7M-parameter token resampler/projector
and its three DeepStack gates are optimized.

Bridge alignment supervises only the first image-classification exchange in
each training record. The later metadata response is intentionally excluded:
under teacher forcing it can read the preceding ground-truth class label,
creating a deceptively low loss without requiring visual alignment.

First run a short bridge alignment pilot. This deliberately writes to a
separate directory and does not read the validation or test split:

```bash
python scripts/train_qwen_bridge.py \
  --output-dir /workspace/qwen-runs/so400m-bridge-pilot20-seed42 \
  --qwen-adapter /workspace/qwen-runs/qwen3-vl-8b-lora-sqrt-balanced-seed42/checkpoint-400 \
  --so400m-checkpoint "$SO400M_CKPT" \
  --max-steps 20 \
  --save-steps 10 \
  --log-steps 1
```

If the loss and gradients remain finite, evaluate the pilot on a balanced
100-image validation subset before launching the full one-epoch run:

```bash
python scripts/evaluate_qwen_bridge.py \
  --data-root /workspace/qwen-data/hyperkvasir \
  --output-dir /workspace/qwen-results/so400m-bridge-pilot20-val100 \
  --qwen-adapter /workspace/qwen-runs/qwen3-vl-8b-lora-sqrt-balanced-seed42/checkpoint-400 \
  --so400m-checkpoint "$SO400M_CKPT" \
  --bridge-checkpoint /workspace/qwen-runs/so400m-bridge-pilot20-seed42/final \
  --split val \
  --limit 100
```

The pilot is a pipeline and learning-signal check, not a selected final model.
The full run must start from a fresh bridge because the pilot uses a shortened
cosine schedule. Interrupted full runs can continue exactly from their latest
optimizer boundary with `--resume-from latest`.

## Claims

Adding Qwen does not by itself make the system a foundation model. A foundation
model claim requires domain-scale pretraining and transfer evaluation across
classification, segmentation, visual question answering, grounding, and ideally
video tasks.
