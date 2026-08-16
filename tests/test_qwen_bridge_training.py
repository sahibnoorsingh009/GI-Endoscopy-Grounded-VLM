from __future__ import annotations

import sys
import tempfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from vlm_demo.qwen_bridge_training import (  # noqa: E402
    LATEST_CHECKPOINT_NAME,
    build_training_plan,
    checkpoint_name,
    epoch_batches,
    resolve_resume_checkpoint,
)
from vlm_demo.qwen_bridge_data import first_assistant_exchange  # noqa: E402


def test_first_assistant_exchange() -> None:
    record = {
        "image": "images/example.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\nClassify."},
            {"from": "gpt", "value": "polyps"},
            {"from": "human", "value": "Give metadata."},
            {"from": "gpt", "value": "GI region: lower."},
        ],
    }
    trimmed = first_assistant_exchange(record)
    assert len(trimmed["conversations"]) == 2
    assert trimmed["conversations"][1]["value"] == "polyps"
    assert len(record["conversations"]) == 4


def test_training_plan() -> None:
    plan = build_training_plan(
        dataset_size=9971,
        batch_size=1,
        gradient_accumulation_steps=8,
        epochs=1,
    )
    assert plan.batches_per_epoch == 9971
    assert plan.optimizer_steps_per_epoch == 1247
    assert plan.total_optimizer_steps == 1247
    limited = build_training_plan(
        dataset_size=9971,
        batch_size=1,
        gradient_accumulation_steps=8,
        epochs=1,
        max_steps=20,
    )
    assert limited.total_optimizer_steps == 20


def test_epoch_batches() -> None:
    first = epoch_batches(dataset_size=11, batch_size=3, seed=42, epoch=0)
    repeated = epoch_batches(dataset_size=11, batch_size=3, seed=42, epoch=0)
    second_epoch = epoch_batches(dataset_size=11, batch_size=3, seed=42, epoch=1)
    assert first == repeated
    assert first != second_epoch
    flattened = [index for batch in first for index in batch]
    assert sorted(flattened) == list(range(11))
    assert [len(batch) for batch in first] == [3, 3, 3, 2]


def test_resume_resolution() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / checkpoint_name(100)
        checkpoint.mkdir()
        for filename in (
            "visual_bridge.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.json",
        ):
            (checkpoint / filename).touch()
        (root / LATEST_CHECKPOINT_NAME).write_text(checkpoint.name + "\n")
        assert resolve_resume_checkpoint(root, "latest") == checkpoint


def main() -> None:
    test_first_assistant_exchange()
    test_training_plan()
    test_epoch_batches()
    test_resume_resolution()
    print("Qwen bridge training contracts: OK")


if __name__ == "__main__":
    main()
