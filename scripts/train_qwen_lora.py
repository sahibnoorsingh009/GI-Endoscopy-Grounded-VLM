#!/usr/bin/env python3
"""Register the local HyperKvasir manifest and start official Qwen3-VL training."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def required_path(name: str, default: str, *, directory: bool = False) -> Path:
    path = Path(os.environ.get(name, default)).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{name} must point to an existing {kind}: {path}")
    return path


def main() -> None:
    framework_root = required_path(
        "QWEN_FINETUNE_ROOT",
        "/workspace/Qwen3-VL/qwen-vl-finetune",
        directory=True,
    )
    annotation_path = required_path(
        "QWEN_TRAIN_JSON",
        "/workspace/qwen-data/hyperkvasir/annotations/hyperkvasir_train.json",
    )
    data_root = required_path(
        "QWEN_DATA_ROOT",
        "/workspace/qwen-data/hyperkvasir",
        directory=True,
    )
    train_module_dir = framework_root / "qwenvl" / "train"
    if not (train_module_dir / "train_qwen.py").is_file():
        raise FileNotFoundError(f"Qwen training module not found under {framework_root}")

    sys.path.insert(0, str(framework_root))
    sys.path.insert(0, str(train_module_dir))

    from qwenvl import data as qwen_data

    qwen_data.data_dict["hyperkvasir_train"] = {
        "annotation_path": str(annotation_path),
        "data_path": str(data_root),
    }

    from qwenvl.train.train_qwen import train

    train(
        attn_implementation=os.environ.get(
            "QWEN_ATTN_IMPLEMENTATION", "flash_attention_2"
        )
    )


if __name__ == "__main__":
    main()
