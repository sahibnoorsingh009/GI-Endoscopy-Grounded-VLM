#!/usr/bin/env python3
"""Create a deterministic capped square-root-balanced Qwen manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path, PurePosixPath


def target_count(count: int, maximum: int, max_repeat: int = 8) -> int:
    if count < 1 or maximum < count or max_repeat < 1:
        raise ValueError("Expected 1 <= count <= maximum and max_repeat >= 1")
    return min(count * max_repeat, math.ceil(math.sqrt(count * maximum)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--max-repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_json.expanduser().resolve()
    index = args.index_csv.expanduser().resolve()
    output = args.output_json.expanduser().resolve()
    summary_path = args.summary_json.expanduser().resolve()
    for path in (output, summary_path):
        if path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --force")

    labels_by_id: dict[str, str] = {}
    with index.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "train":
                raise ValueError("The balancing index must contain only training rows")
            image_id = row["image_id"]
            if image_id in labels_by_id:
                raise ValueError(f"Duplicate training image: {image_id}")
            labels_by_id[image_id] = row["label"]

    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("The source annotation must be a JSON list")
    grouped: dict[str, list[tuple[str, dict[str, object]]]] = defaultdict(list)
    seen: set[str] = set()
    for record in records:
        image = record.get("image")
        if not isinstance(image, str):
            raise TypeError("Every record must contain one image path")
        image_id = PurePosixPath(image).stem
        if image_id in seen or image_id not in labels_by_id:
            raise ValueError(f"Invalid or duplicate training image: {image_id}")
        seen.add(image_id)
        grouped[labels_by_id[image_id]].append((image_id, record))
    if seen != set(labels_by_id):
        raise ValueError("Source annotations and training index do not match")

    maximum = max(len(items) for items in grouped.values())
    balanced: list[dict[str, object]] = []
    class_summary: dict[str, dict[str, float | int]] = {}
    for label in sorted(grouped):
        items = sorted(
            grouped[label],
            key=lambda item: hashlib.sha256(
                f"{args.seed}:{label}:{item[0]}".encode()
            ).digest(),
        )
        count = len(items)
        target = target_count(count, maximum, args.max_repeat)
        balanced.extend(items[index % count][1] for index in range(target))
        class_summary[label] = {
            "source": count,
            "target": target,
            "factor": target / count,
        }

    random.Random(args.seed).shuffle(balanced)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(balanced, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "strategy": "capped_square_root",
        "source_samples": len(records),
        "balanced_samples": len(balanced),
        "max_repeat": args.max_repeat,
        "seed": args.seed,
        "classes": class_summary,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Source samples: {len(records)}")
    print(f"Balanced samples: {len(balanced)}")
    print(f"Annotations: {output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
