from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vlm_demo.qwen_chat import (
    compact_evidence,
    grounded_user_prompt,
    resolve_qwen_adapter,
    text_chat_history,
)


def fixture_evidence() -> dict:
    return {
        "classification": {
            "top_predictions": [
                {
                    "rank": 1,
                    "display_label": "Polyps",
                    "softmax_score": 0.8,
                },
                {
                    "rank": 2,
                    "display_label": "Cecum",
                    "softmax_score": 0.1,
                },
            ],
            "polyp_family_softmax_score": 0.82,
            "score_warning": "Not a calibrated clinical probability.",
        },
        "segmentation": {
            "status": "run",
            "gate_threshold": 0.3,
            "area_percent": 7.25,
            "largest_component_bbox_pixels": (10, 20, 30, 40),
            "largest_component_location": "middle-right",
        },
        "image": {"width": 640, "height": 480},
        "limitations": ["Research prototype."],
    }


def main() -> None:
    compact = compact_evidence(fixture_evidence())
    assert compact["classification"]["top_predictions"][0]["category"] == "Polyps"
    assert compact["segmentation"]["area_percent"] == 7.25
    prompt = grounded_user_prompt("Summarize the image", fixture_evidence())
    assert "Optional specialist-model research context" in prompt
    assert "0.82" in prompt
    assert "LATEST USER MESSAGE" in prompt
    assert prompt.index("Optional specialist-model") < prompt.index(
        "LATEST USER MESSAGE"
    )
    assert prompt.rstrip().endswith("Respond naturally to the latest message.")
    assert grounded_user_prompt("Hello", None) == "Hello"

    contextual_prompt = grounded_user_prompt(
        "What about its location?",
        fixture_evidence(),
        prior_user_questions=["What is the top category?"],
    )
    assert "Earlier user questions" in contextual_prompt
    assert "What about its location?" in contextual_prompt

    history = text_chat_history(
        [
            {"role": "assistant", "content": "Standalone welcome"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "tool", "content": "ignore"},
        ]
    )
    assert history == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]

    with tempfile.TemporaryDirectory() as directory:
        adapter = Path(directory)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"contract-test")
        previous = os.environ.get("QWEN_ADAPTER_PATH")
        os.environ["QWEN_ADAPTER_PATH"] = str(adapter)
        try:
            assert resolve_qwen_adapter() == adapter.resolve()
        finally:
            if previous is None:
                os.environ.pop("QWEN_ADAPTER_PATH", None)
            else:
                os.environ["QWEN_ADAPTER_PATH"] = previous

    print("Qwen grounded-chat contracts: OK")


if __name__ == "__main__":
    main()
