from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel, AutoImageProcessor, AutoModelForImageClassification


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoStageClassifier(nn.Module):
    """Final inference classifier matching the training theory:
    image -> adapted SigLIP2 encoder -> L2 normalize -> saved scaler
          -> Model B coarse MLP
             -> normal leaf: final prediction
             -> merge group: selected Swin fine classifier -> final leaf
    Model A is not part of inference. It is only used during training to discover
    the merge groups.
    """

    def __init__(
        self,
        encoder_checkpoint: str | Path,
        classifier_checkpoint: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.encoder_checkpoint = Path(encoder_checkpoint)
        self.classifier_checkpoint = Path(classifier_checkpoint)

        if not self.encoder_checkpoint.exists():
            raise FileNotFoundError(f"Encoder checkpoint not found: {self.encoder_checkpoint}")
        if not self.classifier_checkpoint.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found: {self.classifier_checkpoint}")

        self.ckpt = torch.load(self.classifier_checkpoint, map_location="cpu", weights_only=False)

        self._load_encoder()
        self._load_scaler()
        self._load_coarse_head()
        self._load_hierarchy()
        self._load_fine_models()
        self.eval()

    def _load_encoder(self) -> None:
        info = self.ckpt.get("encoder", {})
        model_name = info.get("model_name", "google/siglip2-base-patch16-224")
        state_key = info.get("checkpoint_state_key", "student_state_dict")

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        encoder_ckpt = torch.load(self.encoder_checkpoint, map_location="cpu", weights_only=False)
        if state_key not in encoder_ckpt:
            raise KeyError(
                f"Encoder checkpoint does not contain '{state_key}'. "
                f"Available keys: {list(encoder_ckpt.keys())}"
            )

        missing, unexpected = self.encoder.load_state_dict(encoder_ckpt[state_key], strict=False)
        if missing:
            print(f"[encoder] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[encoder] Unexpected keys: {len(unexpected)}")

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.to(self.device).eval()

    def _load_scaler(self) -> None:
        scaler = self.ckpt["scaler"]
    
        self.register_buffer(
            "scaler_mean",
            torch.as_tensor(scaler["mean"], dtype=torch.float32, device=self.device,),
        )
    
        self.register_buffer(
            "scaler_scale",
            torch.as_tensor(scaler["scale"], dtype=torch.float32, device=self.device,),
        )
        
    def _load_coarse_head(self) -> None:
        info = self.ckpt["coarse_head"]
        self.coarse_classes = list(info["classes"])
        self.coarse_head = MLPHead(
            input_dim=int(info["input_dim"]),
            hidden_dim=int(info["hidden_dim"]),
            num_classes=len(self.coarse_classes),
            dropout=float(info["dropout"]),
        )
        self.coarse_head.load_state_dict(info["state_dict"], strict=True)
        self.coarse_head.to(self.device).eval()

    def _load_hierarchy(self) -> None:
        hierarchy = self.ckpt["hierarchy"]
        self.leaf_classes = list(hierarchy.get("leaf_classes", []))
        self.leaf_to_coarse = dict(hierarchy.get("leaf_to_coarse", {}))
        self.merge_groups = list(hierarchy.get("merge_groups", []))

    def _load_fine_models(self) -> None:
        self.fine_models = nn.ModuleDict()
        self.fine_processors: dict[str, Any] = {}
        self.fine_classes: dict[str, list[str]] = {}
        self.merge_to_module_key: dict[str, str] = {}

        for i, (merge_label, info) in enumerate(self.ckpt.get("fine_heads", {}).items()):
            classes = list(info["label_classes"])
            model_name = info["hf_model_name"]
            id2label = {j: str(c) for j, c in enumerate(classes)}
            label2id = {v: k for k, v in id2label.items()}

            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=len(classes),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
            model.load_state_dict(info["model_state_dict"], strict=True)
            model.to(self.device).eval()

            module_key = f"fine_{i}"
            self.fine_models[module_key] = model
            self.fine_processors[merge_label] = processor
            self.fine_classes[merge_label] = classes
            self.merge_to_module_key[merge_label] = module_key

    @torch.no_grad()
    def extract_embedding(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        outputs = self.encoder.get_image_features(**inputs)
        emb = F.normalize(outputs.pooler_output.float(), dim=-1)
        emb = (emb - self.scaler_mean) / self.scaler_scale
        return emb

    @torch.no_grad()
    def predict_coarse(self, image: Image.Image) -> dict[str, Any]:
        logits = self.coarse_head(self.extract_embedding(image))
        probs = F.softmax(logits, dim=1)[0]
    
        idx = int(probs.argmax().item())
    
        return {
            "label": self.coarse_classes[idx],
            "confidence": float(probs[idx].item()),
            "scores": {
                label: float(probs[i].item())
                for i, label in enumerate(self.coarse_classes)
            },
        }


    @torch.no_grad()
    def predict_fine(self, image: Image.Image, merge_label: str) -> dict[str, Any]:
        module_key = self.merge_to_module_key[merge_label]
        model = self.fine_models[module_key]
        processor = self.fine_processors[merge_label]
        classes = self.fine_classes[merge_label]
    
        inputs = processor(
            images=image,
            return_tensors="pt",
        )
    
        outputs = model(
            pixel_values=inputs["pixel_values"].to(self.device)
        )
    
        probs = F.softmax(outputs.logits, dim=1)[0]
    
        idx = int(probs.argmax().item())
    
        return {
            "label": classes[idx],
            "confidence": float(probs[idx].item()),
            "scores": {
                label: float(probs[i].item())
                for i, label in enumerate(classes)
            },
        }

    @torch.no_grad()
    def predict(
        self,
        image: str | Path | Image.Image | np.ndarray,
        top_k: int = 5,
    ):
    
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
    
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
    
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
    
        else:
            raise TypeError("image must be a path, PIL.Image.Image, or numpy.ndarray")

        if self.device.type == "cuda":
            torch.cuda.synchronize()
    
        start = time.perf_counter()
    
        coarse = self.predict_coarse(image)
        coarse_scores = coarse["scores"]
    
        all_scores: dict[str, float] = {}
    
        for coarse_label, coarse_prob in coarse_scores.items():
    
            # Normal leaf -> keep coarse probability directly
            if coarse_label not in self.merge_to_module_key:
                all_scores[coarse_label] = float(coarse_prob)
                continue
    
            # Merge group -> distribute coarse probability
            # across specialist leaf probabilities
            fine = self.predict_fine(
                image=image,
                merge_label=coarse_label,
            )
    
            for fine_label, fine_prob in fine["scores"].items():
                all_scores[fine_label] = (
                    float(coarse_prob) * float(fine_prob)
                )

        ranked = sorted(
            all_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    
        top_k = min(max(int(top_k), 1), len(ranked))
    
        top_predictions = [
            {
                "rank": rank,
                "label": label,
                "display_label": label.replace("-", " ").strip().title(),
                "softmax_score": float(score),
            }
            for rank, (label, score) in enumerate(
                ranked[:top_k],
                start=1,
            )
        ]
    
        if self.device.type == "cuda":
            torch.cuda.synchronize()
    
        classification_seconds = time.perf_counter() - start
    
        return top_predictions, all_scores, classification_seconds
  
    def forward(self, image: str | Path | Image.Image):
        return self.predict(image)



from huggingface_hub import hf_hub_download
def load_classifier(
    encoder_checkpoint: str | Path | None = None,
    classifier_checkpoint: str | Path | None = None,
    device: str | torch.device | None = None,
) -> TwoStageClassifier:

    if encoder_checkpoint is None:
        encoder_checkpoint = hf_hub_download(
            repo_id="Lian70/classifier",
            filename="siglip2_dino_style_best.pt",
        )

    if classifier_checkpoint is None:
        classifier_checkpoint = hf_hub_download(
            repo_id="Lian70/classifier",
            filename="two_stage_classifier.pt",
        )

    return TwoStageClassifier(
        encoder_checkpoint=encoder_checkpoint,
        classifier_checkpoint=classifier_checkpoint,
        device=device,
    )
