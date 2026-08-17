"""Native-Qwen chat service for the GI Endoscopy research interface.

The selected Qwen LoRA remains on Qwen's validated native visual tower. Specialist
SO400M classification and Kvasir-SEG outputs are supplied as structured grounding
context. The unsuccessful fixed-token SO400M bridge is intentionally not used here.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image


DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_ADAPTER_REPO_ID = "Sahibnoor1/gi-endoscopy-grounded-vlm-checkpoints"
DEFAULT_ADAPTER_SUBFOLDER = (
    "qwen3-vl-8b-lora/sqrt-balanced-seed42/checkpoint-400"
)
QWEN_GENERATION_LOCK = threading.Lock()


SYSTEM_PROMPT = """You are GI-EndoFM, a helpful and natural multimodal assistant
with a specialization in gastrointestinal endoscopy. Converse normally: respond to
greetings, answer general questions, follow conversational context, and discuss an
attached image when the user asks about it. Optional classifier and segmentation
outputs may be supplied with an image. Use them when relevant, but ignore them for
unrelated conversation. When citing those outputs, describe them accurately as
research-model evidence rather than confirmed clinical fact. Answer the latest user
message directly and naturally."""


@dataclass(frozen=True)
class QwenChatConfiguration:
    model_id: str
    adapter_path: Path | None
    use_adapter: bool
    attention_implementation: str
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    repetition_penalty: float


def image_to_rgb(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected an RGB image, received shape {array.shape}")
    return Image.fromarray(array[..., :3].astype(np.uint8), mode="RGB")


def compact_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Keep only user-visible evidence that the language model may cite."""

    classification = evidence["classification"]
    segmentation = evidence["segmentation"]
    return {
        "classification": {
            "taxonomy": "HyperKvasir 23-class benchmark",
            "top_predictions": [
                {
                    "rank": item["rank"],
                    "category": item["display_label"],
                    "softmax_score": round(float(item["softmax_score"]), 6),
                }
                for item in classification["top_predictions"]
            ],
            "combined_polyp_family_score": round(
                float(classification["polyp_family_softmax_score"]), 6
            ),
            "score_warning": classification["score_warning"],
        },
        "segmentation": {
            "status": segmentation["status"],
            "gate_threshold": segmentation["gate_threshold"],
            "area_percent": segmentation.get("area_percent"),
            "largest_component_bbox_pixels": segmentation.get(
                "largest_component_bbox_pixels"
            ),
            "largest_component_location": segmentation.get(
                "largest_component_location"
            ),
            "reason": segmentation.get("reason"),
        },
        "image": evidence["image"],
        "limitations": evidence["limitations"],
    }


def grounded_user_prompt(
    question: str,
    evidence: dict[str, Any] | None,
    *,
    prior_user_questions: Sequence[str] | None = None,
) -> str:
    if evidence is None:
        return question.strip()

    payload = json.dumps(
        compact_evidence(evidence),
        ensure_ascii=False,
        sort_keys=True,
    )
    prior = [item.strip() for item in prior_user_questions or [] if item.strip()]
    prior_context = ""
    if prior:
        prior_context = (
            "Earlier user questions, for conversational context:\n- "
            + "\n- ".join(prior[-3:])
            + "\n\n"
        )
    return (
        "Optional specialist-model research context follows. Use it only if it is "
        "relevant to the latest message; otherwise respond normally and ignore it:\n"
        f"{payload}\n\n"
        f"{prior_context}"
        "LATEST USER MESSAGE:\n"
        f"{question.strip()}\n\n"
        "Respond naturally to the latest message."
    )


def text_chat_history(
    history: Sequence[dict[str, Any]] | None,
    *,
    max_messages: int = 8,
) -> list[dict[str, Any]]:
    """Convert Gradio messages into bounded, text-only Qwen history."""

    cleaned: list[dict[str, Any]] = []
    seen_user = False
    for message in history or []:
        role = str(message.get("role", ""))
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        if role == "user":
            seen_user = True
        elif not seen_user:
            # Do not feed the interface's standalone welcome card to Qwen.
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned[-max_messages:]


def _adapter_has_required_files(path: Path) -> bool:
    config_exists = (path / "adapter_config.json").is_file()
    weights_exist = any(
        (path / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )
    return config_exists and weights_exist


def resolve_qwen_adapter() -> Path:
    local_value = os.getenv("QWEN_ADAPTER_PATH")
    if local_value:
        path = Path(local_value).expanduser().resolve()
        if not _adapter_has_required_files(path):
            raise FileNotFoundError(
                "QWEN_ADAPTER_PATH must contain adapter_config.json and "
                f"adapter weights: {path}"
            )
        return path

    repo_id = os.getenv("QWEN_ADAPTER_REPO_ID", DEFAULT_ADAPTER_REPO_ID)
    subfolder = os.getenv(
        "QWEN_ADAPTER_SUBFOLDER", DEFAULT_ADAPTER_SUBFOLDER
    ).strip("/")
    token = os.getenv("HF_TOKEN") or None
    cache_dir = os.getenv("HF_HUB_CACHE") or None
    snapshot_root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[f"{subfolder}/*"],
            token=token,
            cache_dir=cache_dir,
        )
    )
    path = snapshot_root / subfolder
    if not _adapter_has_required_files(path):
        raise FileNotFoundError(
            f"Downloaded adapter is incomplete under {path}. Check "
            "QWEN_ADAPTER_REPO_ID and QWEN_ADAPTER_SUBFOLDER."
        )
    return path


class QwenGIChatService:
    """Lazy-loaded Qwen3-VL chat service with an optional classification LoRA."""

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The Qwen chat layer requires an NVIDIA GPU. The specialist-only "
                "interface can run without it, but GI-EndoFM chat cannot."
            )

        use_adapter = os.getenv("QWEN_CHAT_USE_ADAPTER", "0") == "1"
        self.configuration = QwenChatConfiguration(
            model_id=os.getenv("QWEN_MODEL_ID", DEFAULT_QWEN_MODEL_ID),
            adapter_path=resolve_qwen_adapter() if use_adapter else None,
            use_adapter=use_adapter,
            attention_implementation=os.getenv(
                "QWEN_ATTN_IMPLEMENTATION", "sdpa"
            ),
            max_new_tokens=int(os.getenv("QWEN_MAX_NEW_TOKENS", "256")),
            do_sample=os.getenv("QWEN_DO_SAMPLE", "1") == "1",
            temperature=float(os.getenv("QWEN_TEMPERATURE", "0.4")),
            top_p=float(os.getenv("QWEN_TOP_P", "0.9")),
            repetition_penalty=float(
                os.getenv("QWEN_REPETITION_PENALTY", "1.08")
            ),
        )
        if self.configuration.max_new_tokens <= 0:
            raise ValueError("QWEN_MAX_NEW_TOKENS must be positive")
        if self.configuration.temperature <= 0:
            raise ValueError("QWEN_TEMPERATURE must be positive")
        if not 0 < self.configuration.top_p <= 1:
            raise ValueError("QWEN_TOP_P must be in (0, 1]")
        if self.configuration.repetition_penalty < 1:
            raise ValueError("QWEN_REPETITION_PENALTY must be at least 1")

        self._load_model()

    def _load_model(self) -> None:
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.configuration.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation=self.configuration.attention_implementation,
        )
        if self.configuration.use_adapter:
            self.model = PeftModel.from_pretrained(
                model,
                self.configuration.adapter_path,
                is_trainable=False,
            )
            self.runtime_label = "GI-EndoFM (Qwen3-VL + GI LoRA) ready"
        else:
            self.model = model
            self.runtime_label = "GI-EndoFM multimodal chat ready"
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.processor = AutoProcessor.from_pretrained(
            self.configuration.model_id
        )
        self.device = self.model.device

    @torch.inference_mode()
    def answer(
        self,
        *,
        image: Image.Image | np.ndarray | None,
        question: str,
        evidence: dict[str, Any] | None,
        history: Sequence[dict[str, Any]] | None = None,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            }
        ]
        for previous in text_chat_history(history):
            messages.append(
                {
                    "role": previous["role"],
                    "content": [
                        {"type": "text", "text": previous["content"]}
                    ],
                }
            )

        # A short-lived local file follows the inference path already validated by
        # the Qwen evaluation script. It is deleted immediately after generation.
        with tempfile.TemporaryDirectory(prefix="gi-endofm-") as directory:
            content: list[dict[str, Any]] = []
            if image is not None:
                image_path = Path(directory) / "image.png"
                image_to_rgb(image).save(image_path)
                content.append({"type": "image", "image": image_path.as_uri()})
            content.append(
                {
                    "type": "text",
                    "text": grounded_user_prompt(question, evidence),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            video_metadata = None
            if videos is not None:
                videos, video_metadata = zip(*videos)
                videos = list(videos)
                video_metadata = list(video_metadata)

            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadata,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            ).to(self.device)

            with QWEN_GENERATION_LOCK:
                generation_kwargs = {
                    "max_new_tokens": self.configuration.max_new_tokens,
                    "do_sample": self.configuration.do_sample,
                    "repetition_penalty": self.configuration.repetition_penalty,
                    "use_cache": True,
                }
                if self.configuration.do_sample:
                    generation_kwargs.update(
                        temperature=self.configuration.temperature,
                        top_p=self.configuration.top_p,
                    )
                generated = self.model.generate(**inputs, **generation_kwargs)

        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        answer = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        if not answer:
            raise RuntimeError("GI-EndoFM returned an empty response")
        return answer
