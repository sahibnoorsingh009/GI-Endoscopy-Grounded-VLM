#!/usr/bin/env python3
"""Run one complete specialist-grounded GI-EndoFM chat turn."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from PIL import Image

from vlm_demo.inference import GroundedGIService
from vlm_demo.qwen_chat import QwenGIChatService


EXAMPLE = (
    REPOSITORY_ROOT
    / "demo"
    / "examples"
    / "images"
    / "cju14pxbaoksp0835qzorx6g6.jpg"
)


def main() -> None:
    with Image.open(EXAMPLE) as handle:
        image = handle.convert("RGB")

    specialist = GroundedGIService()
    result = specialist.analyze(image, force_segmentation=True)
    chat = QwenGIChatService()
    answer = chat.answer(
        image=image,
        question="Summarize the model outputs for this image.",
        evidence=result.evidence,
        history=None,
    )
    if not answer.strip():
        raise RuntimeError("GI-EndoFM returned an empty answer")

    print("Top category:", result.top_predictions[0]["display_label"])
    print("GI-EndoFM response:", answer)
    print("Native-Qwen grounded chat smoke test: OK")


if __name__ == "__main__":
    main()
