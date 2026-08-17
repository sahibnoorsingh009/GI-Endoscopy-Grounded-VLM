#!/usr/bin/env python3
"""Distill native-Qwen visual tokens into a frozen-SO400M bridge."""

from __future__ import annotations

import argparse
import json
import math
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

from vlm_demo.qwen_bridge_model import (  # noqa: E402
    SO400MBridgeVisionAdapter,
    load_visual_bridge,
)
from vlm_demo.qwen_bridge_training import (  # noqa: E402
    load_training_checkpoint,
    resolve_resume_checkpoint,
    save_training_checkpoint,
)
from vlm_demo.qwen_distillation import (  # noqa: E402
    QwenBridgeDistillationDataset,
    QwenDistillationCollator,
    build_distillation_plan,
    shard_aware_epoch_batches,
    visual_distillation_losses,
)
from vlm_demo.so400m_encoder import build_frozen_so400m_encoder  # noqa: E402
from vlm_demo.visual_bridge import SO400MVisualBridge, VisualBridgeConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-index",
        type=Path,
        default=Path("/workspace/qwen-data/hyperkvasir/index/hyperkvasir_train.csv"),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("/workspace/qwen-data/hyperkvasir")
    )
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--so400m-checkpoint", type=Path)
    parser.add_argument("--bridge-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from")
    return parser.parse_args()


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if min(
        args.epochs,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.save_steps,
        args.log_steps,
    ) <= 0:
        raise ValueError("Epoch, batch, accumulation, save, and log values must be positive")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0:
        raise ValueError("Learning rate and gradient norm must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if args.bridge_checkpoint is not None and args.resume_from is not None:
        raise ValueError("Use bridge warm start or training resume, not both")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for visual-token distillation")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.exists() and args.resume_from is None:
        raise FileExistsError(
            f"Output directory already contains a run: {config_path}"
        )

    encoder, processor, resolved_so400m = build_frozen_so400m_encoder(
        args.so400m_checkpoint,
        device=device,
        dtype=torch.bfloat16,
    )
    bridge = SO400MVisualBridge(VisualBridgeConfig()).to(
        device=device,
        dtype=torch.bfloat16,
    )
    adapter = SO400MBridgeVisionAdapter(encoder, bridge).to(device)
    adapter.set_bridge_trainable()
    adapter.deepstack_scales.requires_grad = False
    if args.bridge_checkpoint is not None:
        load_visual_bridge(adapter, args.bridge_checkpoint)

    dataset = QwenBridgeDistillationDataset(
        train_index=args.train_index,
        data_root=args.data_root,
        teacher_cache=args.teacher_cache,
        so400m_processor=processor,
        limit=args.limit,
    )
    collator = QwenDistillationCollator()
    plan = build_distillation_plan(
        groups=dataset.shard_index_groups,
        dataset_size=len(dataset),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )
    parameters = [parameter for parameter in bridge.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    warmup_steps = math.ceil(plan.total_optimizer_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=plan.total_optimizer_steps,
    )

    start_epoch = 0
    next_batch_in_epoch = 0
    global_step = 0
    resume = resolve_resume_checkpoint(output_dir, args.resume_from)
    if resume is not None:
        load_visual_bridge(adapter, resume)
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
        if global_step >= plan.total_optimizer_steps:
            raise ValueError("This distillation run is already complete")
        print(f"Resuming from {resume} at step {global_step}")

    run_config = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "objective": {
            "normalized_mse_weight": 1.0,
            "cosine_weight": 1.0,
            "norm_weight": 0.1,
        },
        "training_plan": asdict(plan),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "so400m_checkpoint": str(resolved_so400m),
    }
    if not config_path.exists():
        config_path.write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print("Distillation samples:", len(dataset))
    print("Trainable bridge parameters:", sum(p.numel() for p in parameters))
    print("Optimizer steps:", plan.total_optimizer_steps)
    print("Warmup steps:", warmup_steps)
    print("Output directory:", output_dir)
    adapter.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    windows: dict[str, list[float]] = {
        "loss": [],
        "normalized_mse": [],
        "cosine": [],
        "norm": [],
    }
    finished = False

    for epoch in range(start_epoch, args.epochs):
        all_batches = shard_aware_epoch_batches(
            dataset.shard_index_groups,
            batch_size=args.batch_size,
            seed=args.seed,
            epoch=epoch,
        )
        start_batch = next_batch_in_epoch if epoch == start_epoch else 0
        loader = DataLoader(
            dataset,
            batch_sampler=all_batches[start_batch:],
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        for offset, batch in enumerate(loader, start=start_batch):
            window_start = (
                offset // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps
            window_end = min(
                window_start + args.gradient_accumulation_steps,
                len(all_batches),
            )
            window_size = window_end - window_start
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            teacher = batch["teacher_tokens"].to(device, non_blocking=True)
            with torch.no_grad():
                features = encoder(pixels.to(dtype=torch.bfloat16))
            student = bridge(features.dense_tokens)
            losses = visual_distillation_losses(student, teacher)
            if not all(torch.isfinite(value) for value in losses.values()):
                raise RuntimeError(f"Non-finite distillation loss at step {global_step}: {losses}")
            for name, value in losses.items():
                windows[name].append(float(value.detach().item()))
            (losses["loss"] / window_size).backward()
            if offset + 1 != window_end:
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
                means = {
                    name: sum(values) / len(values)
                    for name, values in windows.items()
                }
                print(
                    f"step={global_step}/{plan.total_optimizer_steps} "
                    f"epoch={epoch + next_batch / len(all_batches):.4f} "
                    f"loss={means['loss']:.6f} mse={means['normalized_mse']:.6f} "
                    f"cosine={means['cosine']:.6f} norm={means['norm']:.6f} "
                    f"grad_norm={float(grad_norm):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e}",
                    flush=True,
                )
                for values in windows.values():
                    values.clear()

            state = {
                "global_step": global_step,
                "epoch": state_epoch,
                "next_batch_in_epoch": next_batch,
                "training_plan": asdict(plan),
            }
            if global_step % args.save_steps == 0:
                saved = save_training_checkpoint(
                    output_dir=output_dir,
                    adapter=adapter,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=state,
                    metadata={
                        "objective": "native_qwen_visual_token_distillation",
                        "teacher_cache": str(args.teacher_cache.expanduser().resolve()),
                        "so400m_checkpoint": str(resolved_so400m),
                    },
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
        "next_batch_in_epoch": next_batch if "next_batch" in locals() else 0,
        "training_plan": asdict(plan),
    }
    final_path = save_training_checkpoint(
        output_dir=output_dir,
        adapter=adapter,
        optimizer=optimizer,
        scheduler=scheduler,
        state=final_state,
        metadata={
            "objective": "native_qwen_visual_token_distillation",
            "teacher_cache": str(args.teacher_cache.expanduser().resolve()),
            "so400m_checkpoint": str(resolved_so400m),
        },
        final=True,
    )
    elapsed = time.perf_counter() - started
    print("Distillation complete")
    print("Global steps:", global_step)
    print("Final checkpoint:", final_path)
    print("Elapsed minutes:", round(elapsed / 60, 2))
    print("Peak GPU memory GiB:", round(torch.cuda.max_memory_allocated() / 1024**3, 3))


if __name__ == "__main__":
    main()
