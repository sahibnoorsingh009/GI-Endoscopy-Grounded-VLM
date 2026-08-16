from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stable_rank(image_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).hexdigest()


def balanced_subset(
    rows: list[dict[str, str]], limit: int | None, seed: int
) -> list[dict[str, str]]:
    if limit is None or limit >= len(rows):
        return sorted(rows, key=lambda row: row["image_id"])
    if limit <= 0:
        raise ValueError("--limit must be positive")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    for label in grouped:
        grouped[label].sort(key=lambda row: stable_rank(row["image_id"], seed))

    selected: list[dict[str, str]] = []
    labels = sorted(grouped)
    offset = 0
    while len(selected) < limit:
        added = False
        for label in labels:
            if offset < len(grouped[label]):
                selected.append(grouped[label][offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def normalize_text(value: str) -> str:
    value = value.strip().casefold().splitlines()[0] if value.strip() else ""
    value = value.strip("`'\" .,:;()[]{}")
    value = re.sub(r"[_\s]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value


def parse_prediction(raw: str, labels: list[str]) -> str | None:
    normalized = normalize_text(raw)
    if normalized in labels:
        return normalized

    aliases = {
        "polyp": "polyps",
        "hemorrhoid": "hemorrhoids",
        "short-segment-barretts": "barretts-short-segment",
    }
    if normalized in aliases and aliases[normalized] in labels:
        return aliases[normalized]

    padded = f"-{normalized}-"
    matches = [
        label
        for label in sorted(labels, key=len, reverse=True)
        if f"-{label}-" in padded
    ]
    return matches[0] if len(matches) == 1 else None


def build_prompt(labels: list[str]) -> str:
    return (
        "Classify this GI endoscopy image using exactly one category from the "
        "following HyperKvasir taxonomy:\n"
        + ", ".join(labels)
        + "\nReturn only the category name and no explanation."
    )


def load_completed(path: Path) -> dict[str, dict[str, object]]:
    completed: dict[str, dict[str, object]] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {error}"
                ) from error
            completed[str(record["image_id"])] = record
    return completed


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def calculate_metrics(
    records: list[dict[str, object]], labels: list[str]
) -> dict[str, object]:
    support = Counter(str(record["ground_truth"]) for record in records)
    predicted = Counter(
        str(record["prediction"])
        for record in records
        if record.get("prediction") is not None
    )
    true_positive = Counter(
        str(record["ground_truth"])
        for record in records
        if record.get("prediction") == record["ground_truth"]
    )

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    recall_values: list[float] = []
    for label in labels:
        tp = true_positive[label]
        actual = support[label]
        predicted_count = predicted[label]
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / actual if actual else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if actual:
            f1_values.append(f1)
            recall_values.append(recall)
        per_class[label] = {
            "support": actual,
            "predicted": predicted_count,
            "true_positive": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    correct = sum(true_positive.values())
    total = len(records)
    exact_raw = sum(bool(record.get("raw_exact_match")) for record in records)
    unparseable = sum(record.get("prediction") is None for record in records)
    return {
        "samples": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "balanced_accuracy": (
            sum(recall_values) / len(recall_values) if recall_values else 0.0
        ),
        "raw_exact_match_rate": exact_raw / total if total else 0.0,
        "unparseable_count": unparseable,
        "unparseable_rate": unparseable / total if total else 0.0,
        "mean_inference_seconds": (
            sum(float(record["inference_seconds"]) for record in records) / total
            if total
            else 0.0
        ),
        "per_class": per_class,
    }


def write_confusion(
    path: Path, records: list[dict[str, object]], labels: list[str]
) -> None:
    matrix = Counter(
        (
            str(record["ground_truth"]),
            str(record["prediction"] or "<unparseable>"),
        )
        for record in records
    )
    prediction_columns = labels + ["<unparseable>"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ground_truth", *prediction_columns])
        for ground_truth in labels:
            writer.writerow(
                [ground_truth]
                + [matrix[(ground_truth, prediction)] for prediction in prediction_columns]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base or LoRA-adapted Qwen3-VL on HyperKvasir."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        help="Optional PEFT LoRA adapter directory to load over the base model.",
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicit confirmation required before reading the frozen test split.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--cleanup-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise ValueError(
            "The test split is frozen. Pass --allow-test only for the final, "
            "pre-declared evaluation."
        )
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = None
    if args.adapter_path is not None:
        adapter_path = args.adapter_path.expanduser().resolve()
        if not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"PEFT adapter_config.json not found under {adapter_path}"
            )

    train_rows = read_csv(data_root / "index" / "hyperkvasir_train.csv")
    evaluation_rows = read_csv(
        data_root / "index" / f"hyperkvasir_{args.split}.csv"
    )
    labels = sorted({row["label"] for row in train_rows})
    if len(labels) != 23:
        raise ValueError(f"Expected 23 training labels, found {len(labels)}")
    selected_rows = balanced_subset(evaluation_rows, args.limit, args.seed)
    selected_ids = {row["image_id"] for row in selected_rows}

    prediction_path = output_dir / "predictions.jsonl"
    if prediction_path.exists() and not args.resume:
        raise FileExistsError(
            f"{prediction_path} already exists; use --resume or another output directory"
        )
    completed = load_completed(prediction_path) if args.resume else {}
    unexpected = set(completed).difference(selected_ids)
    if unexpected:
        raise ValueError(
            "Resume file contains images outside the selected evaluation set: "
            f"{sorted(unexpected)[:5]}"
        )

    prompt = build_prompt(labels)
    processor = AutoProcessor.from_pretrained(args.model_id)
    torch.cuda.reset_peak_memory_stats()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()

    for index, row in enumerate(selected_rows, start=1):
        image_id = row["image_id"]
        if image_id in completed:
            continue
        image_path = (data_root / "images" / f"{image_id}.jpg").resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path.as_uri()},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_metadata = None
        if videos is not None:
            videos, video_metadata = zip(*videos)
            videos = list(videos)
            video_metadata = list(video_metadata)

        inputs = processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        ).to(model.device)

        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        elapsed = time.perf_counter() - start
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        raw_prediction = processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        prediction = parse_prediction(raw_prediction, labels)
        ground_truth = row["label"]
        record: dict[str, object] = {
            "image_id": image_id,
            "ground_truth": ground_truth,
            "raw_prediction": raw_prediction,
            "prediction": prediction,
            "correct": prediction == ground_truth,
            "raw_exact_match": raw_prediction.strip().casefold() == ground_truth,
            "inference_seconds": elapsed,
        }
        append_jsonl(prediction_path, record)
        completed[image_id] = record

        if index == 1 or index % args.report_every == 0 or index == len(selected_rows):
            correct_so_far = sum(
                item.get("prediction") == item["ground_truth"]
                for item in completed.values()
            )
            print(
                f"[{len(completed)}/{len(selected_rows)}] "
                f"accuracy={correct_so_far / len(completed):.4f} "
                f"last={ground_truth}->{prediction or '<unparseable>'} "
                f"time={elapsed:.2f}s",
                flush=True,
            )

        del inputs, generated, new_tokens, images, videos
        if args.cleanup_every and index % args.cleanup_every == 0:
            gc.collect()
            torch.cuda.empty_cache()

    ordered_records = [completed[row["image_id"]] for row in selected_rows]
    metrics = calculate_metrics(ordered_records, labels)
    metrics.update(
        {
            "model_id": args.model_id,
            "adapter_path": str(adapter_path) if adapter_path is not None else None,
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
    write_confusion(output_dir / "confusion_matrix.csv", ordered_records, labels)

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
