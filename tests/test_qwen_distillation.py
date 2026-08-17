from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from vlm_demo.qwen_distillation import (  # noqa: E402
    TeacherTokenStore,
    build_distillation_plan,
    shard_aware_epoch_batches,
    visual_distillation_losses,
)
from vlm_demo.qwen_teacher_cache import (  # noqa: E402
    CACHE_MANIFEST_NAME,
    shard_name,
    write_teacher_shard,
)


def test_store_and_shard_batches() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_teacher_shard(root / shard_name(0), torch.ones(2, 64, 4096), ["a", "b"])
        write_teacher_shard(root / shard_name(1), torch.ones(1, 64, 4096) * 2, ["c"])
        (root / CACHE_MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "complete": True,
                    "cached_samples": 3,
                    "shards": [
                        {"file": shard_name(0)},
                        {"file": shard_name(1)},
                    ],
                }
            )
        )
        store = TeacherTokenStore(root)
        assert store.image_ids == {"a", "b", "c"}
        assert torch.equal(store.get("c"), torch.ones(64, 4096) * 2)

    groups = [[0, 1, 2, 3], [4, 5, 6]]
    first = shard_aware_epoch_batches(groups, batch_size=2, seed=42, epoch=0)
    repeated = shard_aware_epoch_batches(groups, batch_size=2, seed=42, epoch=0)
    assert first == repeated
    assert sorted(index for batch in first for index in batch) == list(range(7))
    assert all(set(batch).issubset(set(group)) for batch in first for group in groups if batch[0] in group)


def test_plan_and_losses() -> None:
    plan = build_distillation_plan(
        groups=[[0, 1, 2, 3], [4, 5, 6]],
        dataset_size=7,
        batch_size=2,
        gradient_accumulation_steps=2,
        epochs=3,
    )
    assert plan.batches_per_epoch == 4
    assert plan.optimizer_steps_per_epoch == 2
    assert plan.total_optimizer_steps == 6

    teacher = torch.randn(2, 64, 4096)
    identical = visual_distillation_losses(teacher.clone(), teacher)
    assert identical["loss"].abs() < 1e-6
    different = visual_distillation_losses(torch.zeros_like(teacher), teacher)
    assert different["loss"] > 0


def main() -> None:
    test_store_and_shard_batches()
    test_plan_and_losses()
    print("Qwen visual-distillation contracts: OK")


if __name__ == "__main__":
    main()
