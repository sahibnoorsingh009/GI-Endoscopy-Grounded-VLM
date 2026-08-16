from __future__ import annotations

import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.build_qwen_manifest import deterministic_prompt, qwen_record  # noqa: E402


class Row:
    image_id = "HK_000000001"
    label = "polyps"
    organ_region = "lower-gi-tract"
    category = "pathological-findings"


def main() -> None:
    first = deterministic_prompt(Row.image_id)
    second = deterministic_prompt(Row.image_id)
    if first != second or not first.startswith("<image>\n"):
        raise SystemExit("Prompt selection is not deterministic or lacks <image>")

    record = qwen_record(Row())
    if record["image"] != "images/HK_000000001.jpg":
        raise SystemExit("Unexpected relative image path")
    conversations = record["conversations"]
    if conversations[1] != {"from": "gpt", "value": "polyps"}:
        raise SystemExit("Canonical label answer changed")
    if sum(
        str(turn["value"]).count("<image>") for turn in conversations
    ) != 1:
        raise SystemExit("Each record must have exactly one <image> tag")

    json.dumps(record)
    print("Qwen manifest builder contract: OK")


if __name__ == "__main__":
    main()
