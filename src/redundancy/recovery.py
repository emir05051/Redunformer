from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from redundancy.pruning import PruningPlan, apply_pruning_plan


@dataclass(frozen=True)
class RecoveryConfig:
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecoveryResult:
    model: Any
    pruning_handles: list[Any]
    training_history: list[dict[str, float | int]]
    trainable_parameters: int
    total_parameters: int
    optimizer_steps: int
    tokens_seen: int


def _is_quantized(model) -> bool:
    return bool(
        getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_loaded_in_8bit", False)
    )


def _prepare_lora_model(model, config: RecoveryConfig):
    if _is_quantized(model):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def recover_with_lora(
    model,
    tokenizer,
    dataset,
    plan: PruningPlan,
    config: RecoveryConfig | None = None,
    device: torch.device | str | None = None,
) -> RecoveryResult:
    del tokenizer  # The packed dataset is already tokenized.
    config = config or RecoveryConfig()
    if len(dataset) == 0:
        raise ValueError("Recovery dataset is empty")
    if config.gradient_accumulation_steps <= 0 or config.micro_batch_size <= 0:
        raise ValueError("Batch sizes and accumulation steps must be positive")

    model = _prepare_lora_model(model, config)
    pruning_handles = apply_pruning_plan(model, plan)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters == 0:
        raise RuntimeError("PEFT did not create any trainable parameters")

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    data_loader = DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer_steps = math.ceil(
        len(data_loader) / config.gradient_accumulation_steps
    )
    warmup_steps = math.ceil(optimizer_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=optimizer_steps,
    )

    original_use_cache = getattr(model.config, "use_cache", None)
    if original_use_cache is not None:
        model.config.use_cache = False
    model.train()
    optimizer.zero_grad(set_to_none=True)
    history: list[dict[str, float | int]] = []
    group_losses: list[float] = []
    examples_seen = 0
    tokens_seen = 0
    completed_steps = 0
    sequence_length = int(dataset.metadata.sequence_length)

    try:
        for batch_index, batch in enumerate(data_loader, start=1):
            batch = {name: value.to(device) for name, value in batch.items()}
            autocast_enabled = device.type == "cuda"
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite recovery loss at batch {batch_index}")
            (loss / config.gradient_accumulation_steps).backward()
            group_losses.append(float(loss.detach().cpu()))
            examples_seen += int(batch["input_ids"].shape[0])
            tokens_seen += int(batch["input_ids"].numel())

            should_step = (
                batch_index % config.gradient_accumulation_steps == 0
                or batch_index == len(data_loader)
            )
            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                config.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed_steps += 1
            history.append(
                {
                    "step": completed_steps,
                    "loss": sum(group_losses) / len(group_losses),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "examples_seen": examples_seen,
                    "tokens_seen": tokens_seen,
                }
            )
            group_losses.clear()
    finally:
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache

    model.eval()
    return RecoveryResult(
        model=model,
        pruning_handles=pruning_handles,
        training_history=history,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        optimizer_steps=completed_steps,
        tokens_seen=tokens_seen,
    )


def save_recovery_adapter(result: RecoveryResult, output_directory: str | Path) -> Path:
    adapter_directory = Path(output_directory) / "adapter"
    adapter_directory.mkdir(parents=True, exist_ok=True)
    result.model.save_pretrained(adapter_directory)
    return adapter_directory


def load_recovered_model(
    base_model,
    adapter_path: str | Path,
    plan_path: str | Path,
    quantization: str = "auto",
):
    """Load an adapter and reapply its pruning plan to a base model or model ID."""
    if quantization not in {"auto", "4bit", "none"}:
        raise ValueError("quantization must be one of: auto, 4bit, none")
    plan = PruningPlan.load(plan_path)
    if isinstance(base_model, str):
        from redundancy.models import RedundancyModel

        base_model = RedundancyModel(
            base_model,
            quantization=quantization,
            revision=plan.model_revision,
        ).model
    model = PeftModel.from_pretrained(base_model, adapter_path)
    handles = apply_pruning_plan(model, plan)
    model.eval()
    return model, handles
