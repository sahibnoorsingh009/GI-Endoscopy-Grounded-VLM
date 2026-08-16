from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluate_qwen_zero_shot import (  # noqa: E402
    balanced_subset,
    calculate_metrics,
    parse_prediction,
)


def main() -> None:
    labels = ["barretts", "barretts-short-segment", "polyps"]
    if parse_prediction("polyps", labels) != "polyps":
        raise SystemExit("Exact parsing failed")
    if parse_prediction("Polyp.", labels) != "polyps":
        raise SystemExit("Alias parsing failed")
    if parse_prediction("barretts-short-segment", labels) != "barretts-short-segment":
        raise SystemExit("Longest-label parsing failed")

    rows = [
        {"image_id": "a1", "label": "a"},
        {"image_id": "a2", "label": "a"},
        {"image_id": "b1", "label": "b"},
        {"image_id": "b2", "label": "b"},
    ]
    selected = balanced_subset(rows, limit=2, seed=42)
    if {row["label"] for row in selected} != {"a", "b"}:
        raise SystemExit("Balanced selection failed")

    records = [
        {
            "ground_truth": "a",
            "prediction": "a",
            "raw_exact_match": True,
            "inference_seconds": 1.0,
        },
        {
            "ground_truth": "b",
            "prediction": "a",
            "raw_exact_match": False,
            "inference_seconds": 3.0,
        },
    ]
    metrics = calculate_metrics(records, ["a", "b"])
    if metrics["accuracy"] != 0.5 or metrics["macro_f1"] != 1.0 / 3.0:
        raise SystemExit(f"Metric calculation failed: {metrics}")
    if metrics["mean_inference_seconds"] != 2.0:
        raise SystemExit("Runtime aggregation failed")

    print("Qwen zero-shot evaluation metrics: OK")


if __name__ == "__main__":
    main()
