from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


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


if __name__ == "__main__":
    main()
