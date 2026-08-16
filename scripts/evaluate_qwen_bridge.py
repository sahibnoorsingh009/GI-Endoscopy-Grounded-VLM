#!/usr/bin/env python3
"""Evaluate a trained SO400M-Qwen bridge on the official validation split."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluate_qwen_zero_shot import (  # noqa: E402
    append_jsonl,
    balanced_subset,
    build_prompt,
    calculate_metrics,
    load_completed,
    parse_prediction,
    read_csv,
    write_confusion,
)
from vlm_demo.qwen_bridge_data import prepare_bridge_inference_example  # noqa: E402
from vlm_demo.qwen_bridge_model import build_qwen_bridge_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bridge-checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-adapter", type=Path, required=True)
    parser.add_argument("--so400m-checkpoint", type=Path)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--cleanup-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise ValueError(
            "The test split is frozen. Pass --allow-test only for the final, "
            "pre-declared evaluation."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_csv(data_root / "index" / "hyperkvasir_train.csv")
    evaluation_rows = read_csv(data_root / "index" / f"hyperkvasir_{args.split}.csv")
    labels = sorted({row["label"] for row in train_rows})
    if len(labels) != 23:
        raise ValueError(f"Expected 23 labels, found {len(labels)}")
    selected_rows = balanced_subset(evaluation_rows, args.limit, args.seed)
    selected_ids = {row["image_id"] for row in selected_rows}

    prediction_path = output_dir / "predictions.jsonl"
    if prediction_path.exists() and not args.resume:
        raise FileExistsError(
            f"{prediction_path} exists; pass --resume or use another output directory"
        )
    completed = load_completed(prediction_path) if args.resume else {}
    unexpected = set(completed).difference(selected_ids)
    if unexpected:
        raise ValueError(f"Resume file has images outside this evaluation: {sorted(unexpected)[:5]}")

    bundle = build_qwen_bridge_bundle(
        qwen_adapter_path=args.qwen_adapter,
        so400m_checkpoint_path=args.so400m_checkpoint,
        bridge_checkpoint_path=args.bridge_checkpoint,
        qwen_model_id=args.model_id,
        device="cuda",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=True,
    )
    bundle.model.eval()
    tokenizer = bundle.qwen_processor.tokenizer
    prompt = build_prompt(labels)
    torch.cuda.reset_peak_memory_stats()

    for index, row in enumerate(selected_rows, start=1):
        image_id = row["image_id"]
        if image_id in completed:
            continue
        record = {
            "image": f"images/{image_id}.jpg",
            "conversations": [
                {"from": "human", "value": f"<image>\n{prompt}"}
            ],
        }
        inputs = prepare_bridge_inference_example(
            record,
            data_root=data_root,
            tokenizer=tokenizer,
            so400m_processor=bundle.so400m_processor,
            image_token_id=bundle.model.config.image_token_id,
            num_queries=bundle.vision_adapter.bridge.config.num_queries,
        )
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = bundle.model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        elapsed = time.perf_counter() - started
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        raw = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        prediction = parse_prediction(raw, labels)
        ground_truth = row["label"]
        result: dict[str, object] = {
            "image_id": image_id,
            "ground_truth": ground_truth,
            "raw_prediction": raw,
            "prediction": prediction,
            "correct": prediction == ground_truth,
            "raw_exact_match": raw.strip().casefold() == ground_truth,
            "inference_seconds": elapsed,
        }
        append_jsonl(prediction_path, result)
        completed[image_id] = result

        if index == 1 or index % args.report_every == 0 or index == len(selected_rows):
            correct = sum(
                item.get("prediction") == item["ground_truth"]
                for item in completed.values()
            )
            print(
                f"[{len(completed)}/{len(selected_rows)}] "
                f"accuracy={correct / len(completed):.4f} "
                f"last={ground_truth}->{prediction or '<unparseable>'} "
                f"time={elapsed:.2f}s",
                flush=True,
            )
        del inputs, generated, new_tokens
        if args.cleanup_every and index % args.cleanup_every == 0:
            gc.collect()
            torch.cuda.empty_cache()

    ordered = [completed[row["image_id"]] for row in selected_rows]
    metrics = calculate_metrics(ordered, labels)
    metrics.update(
        {
            "model_id": args.model_id,
            "qwen_adapter": str(args.qwen_adapter.expanduser().resolve()),
            "bridge_checkpoint": str(args.bridge_checkpoint.expanduser().resolve()),
            "so400m_checkpoint": str(bundle.so400m_checkpoint),
            "split": args.split,
            "selection": "full" if args.limit is None else "balanced_subset",
            "requested_limit": args.limit,
            "seed": args.seed,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_confusion(output_dir / "confusion_matrix.csv", ordered, labels)
    print("\nEvaluation complete")
    print(f"Samples: {metrics['samples']}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Macro F1: {metrics['macro_f1']:.6f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.6f}")
    print(f"Unparseable: {metrics['unparseable_count']}")
    print(f"Mean inference seconds: {metrics['mean_inference_seconds']:.3f}")
    print(f"Peak GPU memory GiB: {metrics['peak_gpu_memory_gib']:.3f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
