from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import vlm_demo.qwen_teacher_cache as cache  # noqa: E402


def test_training_index_policy() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        image_dir = root / "images"
        image_dir.mkdir()
        index = root / "train.csv"
        rows = []
        for value in range(3):
            image_id = f"image-{value}"
            (image_dir / f"{image_id}.jpg").touch()
            rows.append({"image_id": image_id, "split": "train"})
        with index.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("image_id", "split"))
            writer.writeheader()
            writer.writerows(rows)

        original_expected = cache.EXPECTED_TRAIN_IMAGES
        cache.EXPECTED_TRAIN_IMAGES = 3
        try:
            selected = cache.load_training_images(index, root, limit=2)
        finally:
            cache.EXPECTED_TRAIN_IMAGES = original_expected
        assert [item.image_id for item in selected] == ["image-0", "image-1"]


def test_shard_resume_contract() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ids = ["a", "b", "c"]
        cache.write_teacher_shard(
            root / cache.shard_name(0),
            torch.ones(2, 64, 4096),
            ids[:2],
        )
        cache.write_teacher_shard(
            root / cache.shard_name(1),
            torch.ones(1, 64, 4096),
            ids[2:],
        )
        processed, records = cache.inspect_cached_prefix(root, ids)
        assert processed == 3
        assert [record["count"] for record in records] == [2, 1]

        try:
            cache.inspect_cached_prefix(root, ["a", "wrong", "c"])
        except ValueError:
            pass
        else:
            raise AssertionError("A mismatched training prefix should fail")


def main() -> None:
    test_training_index_policy()
    test_shard_resume_contract()
    print("Qwen teacher-cache contracts: OK")


if __name__ == "__main__":
    main()
