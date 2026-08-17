from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


CHECKPOINTS = (
    (
        "classification",
        "Sahibnoor1/gi-siglip2-dino-hyperkvasir-checkpoints",
        "checkpoints/siglip2_so400m_384_supervised_v1/seed42/"
        "so400m_classifier_seed42_vision_ema.pt",
    ),
    (
        "segmentation",
        "Sahibnoor1/kvasir-siglip2-segmentation-checkpoints",
        "checkpoints/siglip2_full/seed_43/best.pt",
    ),
)

QWEN_ADAPTER_REPO_ID = os.getenv(
    "QWEN_ADAPTER_REPO_ID",
    "Sahibnoor1/gi-endoscopy-grounded-vlm-checkpoints",
)
QWEN_ADAPTER_SUBFOLDER = os.getenv(
    "QWEN_ADAPTER_SUBFOLDER",
    "qwen3-vl-8b-lora/sqrt-balanced-seed42/checkpoint-400",
).strip("/")


def main() -> None:
    token = os.getenv("HF_TOKEN")
    for label, repo_id, filename in CHECKPOINTS:
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token or None,
            )
        )
        size_gib = path.stat().st_size / 1024**3
        print(f"{label}: {path} ({size_gib:.2f} GiB)")

    local_adapter = os.getenv("QWEN_ADAPTER_PATH")
    if local_adapter:
        adapter_path = Path(local_adapter).expanduser().resolve()
    else:
        snapshot_root = Path(
            snapshot_download(
                repo_id=QWEN_ADAPTER_REPO_ID,
                repo_type="model",
                allow_patterns=[f"{QWEN_ADAPTER_SUBFOLDER}/*"],
                token=token or None,
            )
        )
        adapter_path = snapshot_root / QWEN_ADAPTER_SUBFOLDER
    adapter_config = adapter_path / "adapter_config.json"
    adapter_weights = next(
        (
            adapter_path / name
            for name in ("adapter_model.safetensors", "adapter_model.bin")
            if (adapter_path / name).is_file()
        ),
        None,
    )
    if not adapter_config.is_file() or adapter_weights is None:
        raise FileNotFoundError(f"Incomplete Qwen adapter under {adapter_path}")
    size_gib = adapter_weights.stat().st_size / 1024**3
    print(f"qwen_adapter: {adapter_path} ({size_gib:.2f} GiB weights)")


if __name__ == "__main__":
    main()
