from __future__ import annotations

import gc
import inspect
import os
import threading
import time
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import AutoConfig, SiglipVisionModel

from src.config import load_config
from src.models import build_model


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

CLASSIFICATION_MODEL_NAME = os.getenv(
    "CLASSIFICATION_MODEL_NAME",
    "google/siglip2-so400m-patch14-384",
)
CLASSIFICATION_REPO_ID = os.getenv(
    "CLASSIFICATION_REPO_ID",
    "Sahibnoor1/gi-siglip2-dino-hyperkvasir-checkpoints",
)
CLASSIFICATION_FILENAME = os.getenv(
    "CLASSIFICATION_CHECKPOINT_FILE",
    "checkpoints/siglip2_so400m_384_supervised_v1/seed42/so400m_classifier_seed42_vision_ema.pt",
)
SEGMENTATION_REPO_ID = os.getenv(
    "SEGMENTATION_REPO_ID",
    "Sahibnoor1/kvasir-siglip2-segmentation-checkpoints",
)
SEGMENTATION_FILENAME = os.getenv(
    "SEGMENTATION_CHECKPOINT_FILE",
    "checkpoints/siglip2_full/seed_43/best.pt",
)
SEGMENTATION_CONFIG = Path(
    os.getenv(
        "SEGMENTATION_CONFIG",
        str(REPOSITORY_ROOT / "configs" / "siglip2_full.yaml"),
    )
)

POLYP_FAMILY = {
    "polyps",
    "dyed-lifted-polyps",
    "dyed-resection-margins",
}
DEFAULT_POLYP_GATE = 0.30
MODEL_LOCK = threading.Lock()


def _torch_load(path: str | Path) -> dict[str, Any]:
    """Load trusted project checkpoints efficiently, including the 12.5 GB archive."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    parameters = inspect.signature(torch.load).parameters
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in parameters:
        kwargs["weights_only"] = True
    if "mmap" in parameters:
        kwargs["mmap"] = True

    try:
        checkpoint = torch.load(checkpoint_path, **kwargs)
    except RuntimeError:
        if "mmap" not in kwargs:
            raise
        kwargs.pop("mmap")
        checkpoint = torch.load(checkpoint_path, **kwargs)

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint {checkpoint_path} is {type(checkpoint).__name__}, not a mapping."
        )
    return checkpoint


def _resolve_checkpoint(
    local_environment_variable: str,
    repo_id: str,
    filename: str,
) -> Path:
    local_value = os.getenv(local_environment_variable)
    if local_value:
        local_path = Path(local_value).expanduser()
        if not local_path.exists():
            raise FileNotFoundError(
                f"{local_environment_variable} points to a missing file: {local_path}"
            )
        return local_path

    download_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "filename": filename,
    }
    token = os.getenv("HF_TOKEN")
    cache_dir = os.getenv("HF_HUB_CACHE")
    if token:
        download_kwargs["token"] = token
    if cache_dir:
        download_kwargs["cache_dir"] = cache_dir
    try:
        return Path(hf_hub_download(**download_kwargs))
    except Exception as error:
        raise RuntimeError(
            f"Could not download {filename} from {repo_id}. If the repository "
            "is private or gated, run `hf auth login` or set an HF_TOKEN with "
            "read access. You can also provide a local checkpoint path through "
            f"{local_environment_variable}."
        ) from error


def _select_classification_state(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], str]:
    for key in ("ema_model", "vision_model", "model", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state, key
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, "<root>"
    raise KeyError(
        "Could not find classification weights. Expected ema_model, vision_model, "
        "model, state_dict, or a raw tensor dictionary."
    )


def _classification_vision_state(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}
    for original_key, value in state_dict.items():
        key = original_key
        while key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("backbone.vision_model."):
            key = "vision_model." + key[len("backbone.vision_model.") :]
        elif key.startswith("encoder.vision_model."):
            key = "vision_model." + key[len("encoder.vision_model.") :]
        elif not key.startswith("vision_model."):
            continue
        mapped[key] = value
    if not mapped:
        raise KeyError("No SigLIP SO400M vision tensors were found in the checkpoint.")
    return mapped


def _classification_head_state(
    checkpoint: dict[str, Any],
    selected_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    source = checkpoint.get("classifier")
    if not isinstance(source, dict):
        source = selected_state

    mapped: dict[str, torch.Tensor] = {}
    for original_key, value in source.items():
        key = original_key
        while key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("classifier."):
            mapped[key[len("classifier.") :]] = value
    if not mapped:
        raise KeyError("No SO400M classification-head tensors were found in the checkpoint.")
    return mapped


def _id_to_label(checkpoint: dict[str, Any]) -> dict[int, str]:
    id2label = checkpoint.get("id2label")
    if isinstance(id2label, dict):
        return {int(index): str(label) for index, label in id2label.items()}

    label2id = checkpoint.get("label2id")
    if isinstance(label2id, dict):
        return {int(index): str(label) for label, index in label2id.items()}

    raise KeyError(
        "Classification checkpoint is missing label2id/id2label metadata. "
        "Use the original best.pt or the repository extraction script."
    )


def _strict_load_vision(
    model: SiglipVisionModel,
    state_dict: dict[str, torch.Tensor],
) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.endswith("position_ids")
    ]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Classification checkpoint does not match SO400M-384. "
            f"Missing: {missing[:20]}; unexpected: {unexpected[:20]}"
        )


def _align_segmentation_encoder_state(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], str | None]:
    """Bridge the direct/wrapped SigLIP2 layouts used by Transformers releases."""
    expected_keys = set(model.state_dict())

    expected_wrapped = any(
        key.startswith("encoder.vision_model.") for key in expected_keys
    )
    checkpoint_wrapped = any(
        key.startswith("encoder.vision_model.") for key in state_dict
    )
    expected_direct = any(key.startswith("encoder.embeddings.") for key in expected_keys)
    checkpoint_direct = any(
        key.startswith("encoder.embeddings.") for key in state_dict
    )

    if expected_wrapped and checkpoint_direct and not checkpoint_wrapped:
        aligned = state_dict.copy()
        for key in tuple(state_dict):
            if not key.startswith("encoder."):
                continue
            new_key = "encoder.vision_model." + key[len("encoder.") :]
            aligned[new_key] = aligned.pop(key)
        return aligned, "direct -> encoder.vision_model wrapper"

    if expected_direct and checkpoint_wrapped and not checkpoint_direct:
        aligned = state_dict.copy()
        for key in tuple(state_dict):
            if not key.startswith("encoder.vision_model."):
                continue
            new_key = "encoder." + key[len("encoder.vision_model.") :]
            aligned[new_key] = aligned.pop(key)
        return aligned, "encoder.vision_model wrapper -> direct"

    return state_dict, None


class SO400MClassifier(nn.Module):
    def __init__(self, model_name: str, number_of_classes: int) -> None:
        super().__init__()
        full_config = AutoConfig.from_pretrained(model_name)
        vision_config = getattr(full_config, "vision_config", full_config)
        self.encoder = SiglipVisionModel(vision_config)
        hidden_size = int(vision_config.hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, number_of_classes),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.encoder(pixel_values=pixel_values, return_dict=True)
        pooled = output.pooler_output
        if pooled is None:
            raise RuntimeError("SO400M did not return its attention-pooled image feature.")
        return self.classifier(pooled)


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    probability: np.ndarray
    elapsed_seconds: float
    area_fraction: float
    largest_component_bbox: tuple[int, int, int, int] | None
    largest_component_centroid: tuple[float, float] | None


@dataclass(frozen=True)
class AnalysisResult:
    original: np.ndarray
    overlay: np.ndarray | None
    mask_image: np.ndarray | None
    top_predictions: list[dict[str, Any]]
    evidence: dict[str, Any]


def _rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Expected an RGB image, received shape {array.shape}.")
    return np.ascontiguousarray(array[..., :3].astype(np.uint8))


def _mask_visual(mask: np.ndarray) -> np.ndarray:
    return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = np.array([0, 220, 100], dtype=np.float32)
    output = image.astype(np.float32).copy()
    foreground = mask.astype(bool)
    output[foreground] = 0.62 * output[foreground] + 0.38 * color
    output = np.clip(output, 0, 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, contours, -1, (100, 220, 0), 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _component_geometry(
    mask: np.ndarray,
) -> tuple[tuple[int, int, int, int] | None, tuple[float, float] | None]:
    count, _, statistics, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return None, None

    largest_label = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    x = int(statistics[largest_label, cv2.CC_STAT_LEFT])
    y = int(statistics[largest_label, cv2.CC_STAT_TOP])
    width = int(statistics[largest_label, cv2.CC_STAT_WIDTH])
    height = int(statistics[largest_label, cv2.CC_STAT_HEIGHT])
    center_x, center_y = centroids[largest_label]
    return (x, y, width, height), (float(center_x), float(center_y))


def _relative_location(
    centroid: tuple[float, float] | None,
    image_width: int,
    image_height: int,
) -> str | None:
    if centroid is None:
        return None
    x_fraction = centroid[0] / max(image_width, 1)
    y_fraction = centroid[1] / max(image_height, 1)
    horizontal = "left" if x_fraction < 1 / 3 else "right" if x_fraction > 2 / 3 else "center"
    vertical = "upper" if y_fraction < 1 / 3 else "lower" if y_fraction > 2 / 3 else "middle"
    return f"{vertical}-{horizontal}"


def _pretty_label(label: str) -> str:
    return label.replace("-", " ").strip().title()


class GroundedGIService:
    """Runs the two validated specialist branches and emits structured evidence."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.polyp_gate_threshold = float(
            os.getenv("POLYP_GATE_THRESHOLD", str(DEFAULT_POLYP_GATE))
        )
        if not 0.0 <= self.polyp_gate_threshold <= 1.0:
            raise ValueError("POLYP_GATE_THRESHOLD must be between 0 and 1.")

        self.classification_checkpoint_path = _resolve_checkpoint(
            "CLASSIFICATION_CHECKPOINT",
            CLASSIFICATION_REPO_ID,
            CLASSIFICATION_FILENAME,
        )
        self.segmentation_checkpoint_path = _resolve_checkpoint(
            "SEGMENTATION_CHECKPOINT",
            SEGMENTATION_REPO_ID,
            SEGMENTATION_FILENAME,
        )

        self._load_classification_model()
        self._load_segmentation_model()

        self.classification_transform = transforms.Compose(
            [
                transforms.Resize(
                    int(384 * 1.15),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def _load_classification_model(self) -> None:
        checkpoint = _torch_load(self.classification_checkpoint_path)
        selected_state, selected_key = _select_classification_state(checkpoint)
        self.id2label = _id_to_label(checkpoint)

        model = SO400MClassifier(
            CLASSIFICATION_MODEL_NAME,
            number_of_classes=len(self.id2label),
        )
        vision_state = _classification_vision_state(selected_state)
        _strict_load_vision(model.encoder, vision_state)
        model.classifier.load_state_dict(
            _classification_head_state(checkpoint, selected_state),
            strict=True,
        )
        self.classification_state_key = selected_key
        self.classification_model = model.eval().to(self.device)

        del vision_state, selected_state, checkpoint
        gc.collect()

    def _load_segmentation_model(self) -> None:
        config = deepcopy(load_config(SEGMENTATION_CONFIG))
        # The fine-tuned checkpoint contains the complete encoder. Build from the
        # small official config instead of downloading redundant base weights.
        config["model"]["load_base_pretrained"] = False
        model = build_model(config)
        checkpoint = _torch_load(self.segmentation_checkpoint_path)
        state_dict = checkpoint.get("model", checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError("Segmentation checkpoint does not contain a model state dictionary.")
        state_dict, layout_change = _align_segmentation_encoder_state(model, state_dict)
        if layout_change:
            print(
                "Adjusted SigLIP2 segmentation encoder checkpoint layout "
                f"({layout_change})."
            )
        model.load_state_dict(state_dict, strict=True)
        self.segmentation_config = config
        self.segmentation_model = model.eval().to(self.device)
        del state_dict, checkpoint
        gc.collect()

    def _autocast(self):
        if self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=self.amp_dtype,
        )

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def classify(
        self,
        image: np.ndarray,
        top_k: int = 5,
    ) -> tuple[list[dict[str, Any]], dict[str, float], float]:
        tensor = self.classification_transform(Image.fromarray(image)).unsqueeze(0)
        tensor = tensor.to(self.device, non_blocking=True)

        self._synchronize()
        start = time.perf_counter()
        with self._autocast():
            logits = self.classification_model(tensor)
        self._synchronize()
        elapsed = time.perf_counter() - start

        probabilities = torch.softmax(logits.float(), dim=1)[0].cpu()
        scores_by_label = {
            self.id2label[index]: float(probabilities[index])
            for index in range(len(probabilities))
        }
        top_count = min(max(int(top_k), 1), len(probabilities))
        top_scores, top_indices = torch.topk(probabilities, k=top_count)
        top_predictions = [
            {
                "rank": rank,
                "label": self.id2label[int(index)],
                "display_label": _pretty_label(self.id2label[int(index)]),
                "softmax_score": float(score),
            }
            for rank, (score, index) in enumerate(
                zip(top_scores.tolist(), top_indices.tolist()),
                start=1,
            )
        ]
        return top_predictions, scores_by_label, elapsed

    @torch.inference_mode()
    def segment(self, image: np.ndarray) -> SegmentationResult:
        height, width = image.shape[:2]
        config = self.segmentation_config
        # Match the checkpoint's training/evaluation geometry exactly. The Kvasir-SEG
        # pipeline resized every image and mask to a square before NaFlex patchifying.
        # Feeding a native-aspect image here creates fewer than 400 valid patches plus
        # padding, which the square decoder cannot spatially reshape without unpadding.
        model_image_size = int(config["data"].get("image_size", 320))
        model_image = cv2.resize(
            image,
            (model_image_size, model_image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        processed = self.segmentation_model.processor(
            images=[Image.fromarray(model_image)],
            return_tensors="pt",
            max_num_patches=int(config["model"].get("max_num_patches", 400)),
        )
        kwargs: dict[str, Any] = {
            "pixel_values": processed["pixel_values"].to(
                self.device, non_blocking=True
            ),
            "output_size": (model_image_size, model_image_size),
        }
        for key in ("pixel_attention_mask", "spatial_shapes"):
            value = processed.get(key)
            if value is not None:
                kwargs[key] = value.to(self.device, non_blocking=True)

        self._synchronize()
        start = time.perf_counter()
        with self._autocast():
            logits = self.segmentation_model(**kwargs)
        if logits.shape[-2:] != (model_image_size, model_image_size):
            logits = torch.nn.functional.interpolate(
                logits,
                size=(model_image_size, model_image_size),
                mode="bilinear",
                align_corners=False,
            )
        if (height, width) != (model_image_size, model_image_size):
            logits = torch.nn.functional.interpolate(
                logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        self._synchronize()
        elapsed = time.perf_counter() - start

        probability = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        threshold = float(config["training"].get("threshold", 0.5))
        mask = (probability >= threshold).astype(np.uint8)
        bbox, centroid = _component_geometry(mask)
        return SegmentationResult(
            mask=mask,
            probability=probability,
            elapsed_seconds=elapsed,
            area_fraction=float(mask.mean()),
            largest_component_bbox=bbox,
            largest_component_centroid=centroid,
        )

    def analyze(
        self,
        image_input: Image.Image | np.ndarray,
        force_segmentation: bool = False,
    ) -> AnalysisResult:
        image = _rgb_array(image_input)
        with MODEL_LOCK:
            top_predictions, all_scores, classification_seconds = self.classify(image)
            polyp_family_score = float(
                sum(all_scores.get(label, 0.0) for label in POLYP_FAMILY)
            )
            segmentation_allowed = (
                force_segmentation
                or polyp_family_score >= self.polyp_gate_threshold
            )
            segmentation = self.segment(image) if segmentation_allowed else None

        height, width = image.shape[:2]
        segmentation_evidence: dict[str, Any]
        overlay_image: np.ndarray | None
        mask_image: np.ndarray | None

        if segmentation is None:
            overlay_image = None
            mask_image = None
            segmentation_evidence = {
                "status": "skipped_by_polyp_gate",
                "reason": (
                    "The Kvasir-SEG model was trained on polyp images and is not a "
                    "validated no-polyp detector."
                ),
                "gate_threshold": self.polyp_gate_threshold,
                "area_fraction": None,
                "largest_component_bbox_pixels": None,
                "largest_component_location": None,
                "elapsed_seconds": None,
            }
        else:
            overlay_image = _overlay(image, segmentation.mask)
            mask_image = _mask_visual(segmentation.mask)
            location = _relative_location(
                segmentation.largest_component_centroid,
                width,
                height,
            )
            segmentation_evidence = {
                "status": "forced" if force_segmentation else "run",
                "gate_threshold": self.polyp_gate_threshold,
                "area_fraction": segmentation.area_fraction,
                "area_percent": segmentation.area_fraction * 100.0,
                "largest_component_bbox_pixels": segmentation.largest_component_bbox,
                "largest_component_location": location,
                "elapsed_seconds": segmentation.elapsed_seconds,
                "checkpoint": SEGMENTATION_FILENAME,
            }

        evidence = {
            "task": "research_only_gi_endoscopy_image_analysis",
            "classification": {
                "top_predictions": top_predictions,
                "polyp_family_labels": sorted(POLYP_FAMILY),
                "polyp_family_softmax_score": polyp_family_score,
                "elapsed_seconds": classification_seconds,
                "score_warning": (
                    "Softmax scores are not calibrated clinical probabilities."
                ),
                "model": CLASSIFICATION_MODEL_NAME,
                "checkpoint": CLASSIFICATION_FILENAME,
                "state_key": self.classification_state_key,
            },
            "segmentation": segmentation_evidence,
            "image": {"width": width, "height": height},
            "limitations": [
                "Research prototype; not validated for diagnosis or patient management.",
                "Classification is limited to the 23 HyperKvasir benchmark categories.",
                "Segmentation was trained on Kvasir-SEG polyp images and is gated on non-polyp inputs.",
            ],
        }
        return AnalysisResult(
            original=image,
            overlay=overlay_image,
            mask_image=mask_image,
            top_predictions=top_predictions,
            evidence=evidence,
        )


def grounded_answer(question: str, evidence: dict[str, Any] | None) -> str:
    """Answer only from specialist-model evidence; no unsupported medical claims."""
    if not evidence:
        return "Run the image analysis first, then ask a question about its model outputs."

    query = (question or "summarize the result").strip().lower()
    classification = evidence["classification"]
    segmentation = evidence["segmentation"]
    predictions = classification["top_predictions"]
    best = predictions[0]
    top_three = ", ".join(
        f"{item['display_label']} ({item['softmax_score']:.1%})"
        for item in predictions[:3]
    )
    polyp_score = classification["polyp_family_softmax_score"]

    clinical_terms = {
        "diagnose",
        "diagnosis",
        "cancer",
        "malignant",
        "benign",
        "treatment",
        "biopsy",
        "patient",
    }
    if any(term in query for term in clinical_terms):
        return (
            "This research interface cannot make a diagnosis, determine malignancy, "
            "or recommend treatment. Its supported output is limited to the trained "
            "HyperKvasir category scores and, when gated on, a Kvasir-SEG polyp mask."
        )

    if any(term in query for term in ("where", "location", "mask", "segment", "area", "size")):
        if segmentation["status"] == "skipped_by_polyp_gate":
            return (
                f"Segmentation was not run because the combined polyp-family score "
                f"was {polyp_score:.1%}, below the {segmentation['gate_threshold']:.0%} "
                "gate. The Kvasir-SEG model is not validated as a no-polyp detector."
            )
        bbox = segmentation["largest_component_bbox_pixels"]
        if bbox is None:
            return (
                "The segmentation branch ran, but no pixels crossed its mask threshold. "
                "That is a model output, not evidence that a lesion is absent."
            )
        x, y, width, height = bbox
        return (
            f"The largest predicted region is in the "
            f"{segmentation['largest_component_location']} part of the image. "
            f"Its bounding box is x={x}, y={y}, width={width}, height={height} pixels, "
            f"and the complete predicted mask covers {segmentation['area_percent']:.2f}% "
            "of the image."
        )

    if any(term in query for term in ("confidence", "sure", "uncertain", "probability", "score")):
        margin = (
            predictions[0]["softmax_score"] - predictions[1]["softmax_score"]
            if len(predictions) > 1
            else predictions[0]["softmax_score"]
        )
        return (
            f"The top softmax score is {best['softmax_score']:.1%} for "
            f"{best['display_label']}; the margin over rank 2 is {margin:.1%}. "
            f"The combined polyp-family score is {polyp_score:.1%}. These scores are "
            "not calibrated clinical probabilities."
        )

    if "polyp" in query:
        gate_status = (
            "passed, so conditional segmentation ran"
            if segmentation["status"] != "skipped_by_polyp_gate"
            else "did not pass, so segmentation was skipped"
        )
        return (
            f"The combined softmax score for the three polyp-family benchmark "
            f"categories is {polyp_score:.1%}. The {segmentation['gate_threshold']:.0%} "
            f"research gate {gate_status}. This does not establish that a polyp is "
            "present or absent."
        )

    if any(term in query for term in ("class", "category", "finding", "predict", "top")):
        return (
            f"The top HyperKvasir benchmark category is {best['display_label']} "
            f"({best['softmax_score']:.1%}). The top three are: {top_three}. "
            "This is a benchmark classification output, not a clinical diagnosis."
        )

    segmentation_summary = (
        "Segmentation was skipped by the polyp gate."
        if segmentation["status"] == "skipped_by_polyp_gate"
        else (
            f"The conditional mask covers {segmentation['area_percent']:.2f}% of the "
            f"image and its largest component is {segmentation['largest_component_location']}."
            if segmentation["largest_component_bbox_pixels"] is not None
            else "The segmentation branch ran but returned an empty thresholded mask."
        )
    )
    return (
        f"Top benchmark category: {best['display_label']} "
        f"({best['softmax_score']:.1%}). Combined polyp-family score: "
        f"{polyp_score:.1%}. {segmentation_summary} Research use only; these outputs "
        "are not validated for clinical decisions."
    )
