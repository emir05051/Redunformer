from __future__ import annotations

import random

from redundancy.pruning.plan import PruningPlan, apply_pruning_plan
from redundancy.utils import get_model_metadata


def select_random_pruning_plan(
    model,
    ratio: float,
    seed: int = 42,
    model_name: str | None = None,
    model_revision: str | None = None,
    **_,
) -> PruningPlan:
    if not 0 <= ratio <= 1:
        raise ValueError("ratio must be between 0 and 1")
    model_metadata = get_model_metadata(model)
    heads_per_layer = max(1, int(model_metadata.num_heads * ratio)) if ratio > 0 else 0
    generator = random.Random(seed)
    selected_heads = {
        layer: sorted(generator.sample(range(model_metadata.num_heads), heads_per_layer))
        for layer in model_metadata.eligible_layers
    }
    return PruningPlan(
        method="random",
        model_name=model_name or getattr(model.config, "_name_or_path", type(model).__name__),
        model_revision=model_revision,
        requested_ratio=ratio,
        actual_ratio=heads_per_layer / model_metadata.num_heads,
        seed=seed,
        eligible_layers=list(model_metadata.eligible_layers),
        selected_heads=selected_heads,
        num_heads=model_metadata.num_heads,
        head_dim=model_metadata.head_dim,
    )


def random_prune_model(model, sparsity: float, seed: int = 42):
    """Backward-compatible helper returning active pruning hooks."""
    plan = select_random_pruning_plan(model=model, ratio=sparsity, seed=seed)
    return apply_pruning_plan(model, plan)


def prune_model(model, sparsity, seed=42):
    return random_prune_model(model=model, sparsity=sparsity, seed=seed)
