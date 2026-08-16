from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = (
    "app.py",
    "requirements.txt",
    "configs/siglip2_full.yaml",
    "vlm_demo/app.py",
    "vlm_demo/inference.py",
    "src/models/siglip2_unet.py",
    "scripts/setup_local.sh",
    "scripts/setup_windows.ps1",
    "scripts/setup_runpod.sh",
    "scripts/build_qwen_manifest.py",
    "scripts/evaluate_qwen_zero_shot.py",
    "docs/QWEN_INTEGRATION.md",
    "requirements-qwen.txt",
)
FORBIDDEN_WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx"}


def main() -> None:
    missing = [path for path in REQUIRED_PATHS if not (REPOSITORY_ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"Missing required repository files: {missing}")

    committed_weights = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in REPOSITORY_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_WEIGHT_SUFFIXES
    ]
    if committed_weights:
        raise SystemExit(f"Model weights must not be committed: {committed_weights}")

    print("Repository structure: OK")


if __name__ == "__main__":
    main()
