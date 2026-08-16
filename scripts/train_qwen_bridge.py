#!/usr/bin/env python3
"""Train only the SO400M-to-Qwen visual bridge on the official train split."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from vlm_demo.qwen_bridge_data import QwenBridgeCollator, QwenBridgeDataset  # noqa: E402
from vlm_demo.qwen_bridge_model import build_qwen_bridge_bundle  # noqa: E402
from vlm_demo.qwen_bridge_training import (  # noqa: E402
    build_training_plan,
    epoch_batches,
    load_training_checkpoint,
    resolve_resume_checkpoint,
    save_training_checkpoint,
)


DEFAULT_ADAPTER = Path(
    "/workspace/qwen-runs/qwen3-vl-8b-lora-sqrt-balanced-seed42/checkpoint-400"
)
DEFAULT_TRAIN_JSON = Path(
    "/workspace/qwen-data/hyperkvasir/annotations/"
    "hyperkvasir_train_sqrt_balanced.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument(
        "--data-root", type=Path, default=Path("/workspace/qwen-data/hyperkvasir")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--so400m-checkpoint", type=Path)
    parser.add_argument("--bridge-checkpoint", type=Path)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from",
        help="Checkpoint directory/name, or 'latest' under --output-dir.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    for name in ("learning_rate", "max_grad_norm"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("save_steps", "log_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def checkpoint_metadata(
    args: argparse.Namespace,
    dataset_size: int,
    resolved_so400m_checkpoint: Path,
) -> dict[str, object]:
    return {
        "research_use_only": True,
        "qwen_model_id": args.model_id,
        "qwen_adapter": str(args.qwen_adapter.expanduser().resolve()),
        "so400m_checkpoint": str(resolved_so400m_checkpoint),
        "train_annotation": str(args.train_json.expanduser().resolve()),
        "train_samples": dataset_size,
        "seed": args.seed,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.bridge_checkpoint is not None and args.resume_from is not None:
        raise ValueError(
            "Use --bridge-checkpoint for a new warm-started run or "
            "--resume-from for an interrupted run, not both."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen bridge training")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.exists() and args.resume_from is None:
        raise FileExistsError(
            f"Output directory already contains a run: {config_path}. "
            "Use a new directory or --resume-from latest."
        )

    bundle = build_qwen_bridge_bundle(
        qwen_adapter_path=args.qwen_adapter,
        so400m_checkpoint_path=args.so400m_checkpoint,
        bridge_checkpoint_path=args.bridge_checkpoint,
        qwen_model_id=args.model_id,
        device=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    tokenizer = bundle.qwen_processor.tokenizer
    dataset = QwenBridgeDataset(
        args.train_json,
        args.data_root,
        tokenizer=tokenizer,
        so400m_processor=bundle.so400m_processor,
        image_token_id=bundle.model.config.image_token_id,
        num_queries=bundle.vision_adapter.bridge.config.num_queries,
        max_length=args.max_length,
        first_exchange_only=True,
    )
    collator = QwenBridgeCollator(tokenizer.pad_token_id)
    plan = build_training_plan(
        dataset_size=len(dataset),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )
    parameters = bundle.vision_adapter.trainable_parameters()
    optimizer_kwargs: dict[str, object] = {
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    if "fused" in torch.optim.AdamW.__init__.__code__.co_varnames:
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(parameters, **optimizer_kwargs)
    warmup_steps = math.ceil(plan.total_optimizer_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=plan.total_optimizer_steps,
    )

    resume = resolve_resume_checkpoint(output_dir, args.resume_from)
    start_epoch = 0
    next_batch_in_epoch = 0
    global_step = 0
    if resume is not None:
        from vlm_demo.qwen_bridge_model import load_visual_bridge

        load_visual_bridge(bundle.vision_adapter, resume)
        state = load_training_checkpoint(
            checkpoint=resume,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_plan=plan,
        )
        move_optimizer_state(optimizer, device)
        start_epoch = int(state["epoch"])
        next_batch_in_epoch = int(state["next_batch_in_epoch"])
        global_step = int(state["global_step"])
        print(f"Resuming from {resume} at optimizer step {global_step}")
        if global_step >= plan.total_optimizer_steps:
            raise ValueError(
                "This training plan is already complete. Start a new output "
                "directory for another experiment."
            )

    run_config = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "training_plan": asdict(plan),
        "trainable_parameters": bundle.vision_adapter.trainable_parameter_count,
        "so400m_checkpoint": str(bundle.so400m_checkpoint),
        "training_objective": "first_image_classification_exchange_only",
    }
    if not config_path.exists():
        config_path.write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print("Train samples:", len(dataset))
    print("Trainable bridge parameters:", bundle.vision_adapter.trainable_parameter_count)
    print("Optimizer steps:", plan.total_optimizer_steps)
    print("Warmup steps:", warmup_steps)
    print("Output directory:", output_dir)

    bundle.model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss_window: list[float] = []
    finished = global_step >= plan.total_optimizer_steps

    for epoch in range(start_epoch, args.epochs):
        all_batches = epoch_batches(
            dataset_size=len(dataset),
            batch_size=args.batch_size,
            seed=args.seed,
            epoch=epoch,
        )
        start_batch = next_batch_in_epoch if epoch == start_epoch else 0
        remaining_batches = all_batches[start_batch:]
        loader = DataLoader(
            dataset,
            batch_sampler=remaining_batches,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        for offset, batch in enumerate(loader, start=start_batch):
            window_start = (offset // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            window_end = min(
                window_start + args.gradient_accumulation_steps,
                len(all_batches),
            )
            window_size = window_end - window_start
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            outputs = bundle.model(**batch)
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {offset}: {loss}")
            loss_window.append(float(loss.detach().item()))
            (loss / window_size).backward()
            should_step = offset + 1 == window_end
            if not should_step:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm: {grad_norm}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            next_batch = offset + 1
            state_epoch = epoch
            if next_batch >= len(all_batches):
                state_epoch = epoch + 1
                next_batch = 0

            if global_step == 1 or global_step % args.log_steps == 0:
                elapsed = time.perf_counter() - started
                mean_loss = sum(loss_window) / len(loss_window)
                print(
                    f"step={global_step}/{plan.total_optimizer_steps} "
                    f"epoch={epoch + next_batch / len(all_batches):.4f} "
                    f"loss={mean_loss:.6f} grad_norm={float(grad_norm):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} "
                    f"elapsed_min={elapsed / 60:.1f}",
                    flush=True,
                )
                loss_window.clear()

            state = {
                "global_step": global_step,
                "epoch": state_epoch,
                "next_batch_in_epoch": next_batch,
                "training_plan": asdict(plan),
            }
            if global_step % args.save_steps == 0:
                saved = save_training_checkpoint(
                    output_dir=output_dir,
                    adapter=bundle.vision_adapter,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=state,
                    metadata=checkpoint_metadata(
                        args,
                        len(dataset),
                        bundle.so400m_checkpoint,
                    ),
                )
                print("Saved:", saved, flush=True)

            if global_step >= plan.total_optimizer_steps:
                finished = True
                break

        next_batch_in_epoch = 0
        if finished:
            break

    final_state = {
        "global_step": global_step,
        "epoch": state_epoch if "state_epoch" in locals() else start_epoch,
        "next_batch_in_epoch": next_batch if "next_batch" in locals() else next_batch_in_epoch,
        "training_plan": asdict(plan),
    }
    final_path = save_training_checkpoint(
        output_dir=output_dir,
        adapter=bundle.vision_adapter,
        optimizer=optimizer,
        scheduler=scheduler,
        state=final_state,
        metadata=checkpoint_metadata(
            args,
            len(dataset),
            bundle.so400m_checkpoint,
        ),
        final=True,
    )
    elapsed = time.perf_counter() - started
    print("Training complete")
    print("Global steps:", global_step)
    print("Final checkpoint:", final_path)
    print("Elapsed minutes:", round(elapsed / 60, 2))
    print("Peak GPU memory GiB:", round(torch.cuda.max_memory_allocated() / 1024**3, 3))


if __name__ == "__main__":
    main()
