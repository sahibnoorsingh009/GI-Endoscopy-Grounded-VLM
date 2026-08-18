from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from PIL import Image

from vlm_demo.inference_new import GroundedGIService


EXAMPLE = (
    REPOSITORY_ROOT
    / "demo"
    / "examples"
    / "images"
    / "cju14pxbaoksp0835qzorx6g6.jpg"
)


def main() -> None:
    service = GroundedGIService()
    with Image.open(EXAMPLE) as image:
        result = service.analyze(image.convert("RGB"), force_segmentation=True)

    assert result.original.shape[:2] == result.mask_image.shape[:2]
    assert result.overlay is not None
    assert result.top_predictions
    print(f"Top category: {result.top_predictions[0]['display_label']}")
    print(
        "Mask area: "
        f"{result.evidence['segmentation']['area_percent']:.2f}% of the image"
    )
    print("End-to-end smoke test: OK")


if __name__ == "__main__":
    main()
