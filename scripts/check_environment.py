from __future__ import annotations

import platform
import sys

import gradio
import huggingface_hub
import peft
import torch
import transformers


def main() -> None:
    print(f"Python: {sys.version.split()[0]} ({platform.system()})")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Gradio: {gradio.__version__}")
    print(f"Hugging Face Hub: {huggingface_hub.__version__}")
    print(f"PEFT: {peft.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory: {memory_gib:.1f} GiB")
    else:
        print("Warning: CPU execution is supported but is not recommended for this demo.")

    from transformers import (  # noqa: F401
        Qwen3VLForConditionalGeneration,
        Siglip2VisionModel,
        SiglipVisionModel,
    )
    from qwen_vl_utils import process_vision_info  # noqa: F401

    print("SigLIP, SigLIP2, and Qwen3-VL imports: OK")


if __name__ == "__main__":
    main()
