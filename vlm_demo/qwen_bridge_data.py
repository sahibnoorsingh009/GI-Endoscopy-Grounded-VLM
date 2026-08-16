"""Data contract for replacing Qwen image pads with 64 SO400M bridge tokens."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset


IGNORE_INDEX = -100


def annotation_to_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    image_markers = 0

    for turn in record["conversations"]:
        value = str(turn["value"])
        if turn["from"] == "human":
            content: list[dict[str, str]] = []
            for part in re.split(r"(<image>)", value):
                if part == "<image>":
                    content.append({"type": "image"})
                    image_markers += 1
                elif part.strip():
                    content.append({"type": "text", "text": part.strip()})
            messages.append({"role": "user", "content": content})
        elif turn["from"] == "gpt":
            messages.append({"role": "assistant", "content": value})
        else:
            raise ValueError(f"Unsupported conversation role: {turn['from']!r}")

    if image_markers != 1:
        raise ValueError(
            f"Each bridge example must contain one <image>, found {image_markers}"
        )
    return messages


def expand_image_tokens(
    input_ids: torch.Tensor,
    *,
    image_token_id: int,
    num_queries: int,
) -> torch.Tensor:
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be rank 1, got {tuple(input_ids.shape)}")
    positions = torch.nonzero(input_ids == image_token_id, as_tuple=False).flatten()
    if positions.numel() != 1:
        raise ValueError(
            f"Expected one image placeholder token, found {positions.numel()}"
        )
    if num_queries <= 0:
        raise ValueError("num_queries must be positive")

    position = int(positions.item())
    repeated = input_ids.new_full((num_queries,), image_token_id)
    return torch.cat(
        (input_ids[:position], repeated, input_ids[position + 1 :]),
        dim=0,
    )


def build_assistant_labels(
    input_ids: torch.Tensor,
    tokenizer: Any,
) -> torch.Tensor:
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_role = tokenizer.encode("assistant\n", add_special_tokens=False)
    prefix = [im_start_id, *assistant_role]
    values = input_ids.tolist()

    supervised_spans = 0
    index = 0
    while index <= len(values) - len(prefix):
        if values[index : index + len(prefix)] != prefix:
            index += 1
            continue

        answer_start = index + len(prefix)
        try:
            answer_end = values.index(im_end_id, answer_start)
        except ValueError as error:
            raise ValueError("Assistant response is missing <|im_end|>") from error
        labels[answer_start : answer_end + 1] = input_ids[
            answer_start : answer_end + 1
        ]
        supervised_spans += 1
        index = answer_end + 1

    if supervised_spans == 0 or not torch.any(labels != IGNORE_INDEX):
        raise ValueError("No supervised assistant tokens were found")
    return labels


def image_grid_thw(num_queries: int, spatial_merge_size: int = 2) -> torch.Tensor:
    side = math.isqrt(num_queries)
    if side * side != num_queries:
        raise ValueError("num_queries must form a square token grid")
    patch_side = side * spatial_merge_size
    return torch.tensor([1, patch_side, patch_side], dtype=torch.long)


def tokenize_bridge_conversation(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    image_token_id: int,
    num_queries: int,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    messages = annotation_to_messages(record)
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if not torch.is_tensor(encoded):
        encoded = torch.tensor(encoded, dtype=torch.long)
    input_ids = encoded.squeeze(0)
    input_ids = expand_image_tokens(
        input_ids,
        image_token_id=image_token_id,
        num_queries=num_queries,
    )
    if input_ids.numel() > max_length:
        raise ValueError(
            f"Bridge example has {input_ids.numel()} tokens, exceeding {max_length}"
        )
    labels = build_assistant_labels(input_ids, tokenizer)
    return input_ids, labels


def resolve_record_image(record: dict[str, Any], data_root: Path) -> Path:
    image = record.get("image")
    if not isinstance(image, str):
        raise ValueError("Bridge annotations must contain one string image path")
    path = (data_root / image).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def prepare_bridge_example(
    record: dict[str, Any],
    *,
    data_root: Path,
    tokenizer: Any,
    so400m_processor: Any,
    image_token_id: int,
    num_queries: int = 64,
    max_length: int = 512,
) -> dict[str, torch.Tensor]:
    input_ids, labels = tokenize_bridge_conversation(
        record,
        tokenizer,
        image_token_id=image_token_id,
        num_queries=num_queries,
        max_length=max_length,
    )
    image_path = resolve_record_image(record, data_root)
    with Image.open(image_path) as image_handle:
        image = image_handle.convert("RGB")
    pixel_values = so400m_processor(
        images=image,
        return_tensors="pt",
    )["pixel_values"].squeeze(0)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw(num_queries),
    }


class QwenBridgeDataset(Dataset):
    def __init__(
        self,
        annotation_path: str | Path,
        data_root: str | Path,
        *,
        tokenizer: Any,
        so400m_processor: Any,
        image_token_id: int,
        num_queries: int = 64,
        max_length: int = 512,
    ) -> None:
        self.annotation_path = Path(annotation_path).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.annotation_path.is_file():
            raise FileNotFoundError(self.annotation_path)
        if not self.data_root.is_dir():
            raise FileNotFoundError(self.data_root)

        self.records = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        if not isinstance(self.records, list) or not self.records:
            raise ValueError("Bridge annotation must be a non-empty JSON list")
        self.tokenizer = tokenizer
        self.so400m_processor = so400m_processor
        self.image_token_id = image_token_id
        self.num_queries = num_queries
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return prepare_bridge_example(
            self.records[index],
            data_root=self.data_root,
            tokenizer=self.tokenizer,
            so400m_processor=self.so400m_processor,
            image_token_id=self.image_token_id,
            num_queries=self.num_queries,
            max_length=self.max_length,
        )


class QwenBridgeCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        examples: Sequence[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [example["input_ids"] for example in examples],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [example["labels"] for example in examples],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.pad_token_id),
            "pixel_values": torch.stack(
                [example["pixel_values"] for example in examples]
            ),
            "image_grid_thw": torch.stack(
                [example["image_grid_thw"] for example in examples]
            ),
        }
