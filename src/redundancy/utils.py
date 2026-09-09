from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelMetadata:
    model_type: str
    num_layers: int
    num_heads: int
    head_dim: int
    eligible_layers: tuple[int, ...]
    max_position_embeddings: int


def _text_config(config: Any) -> Any:
    return getattr(config, "text_config", None) or config


def get_model_metadata(model) -> ModelMetadata:
    config = model.config
    text_config = _text_config(config)
    model_type = str(
        getattr(text_config, "model_type", getattr(config, "model_type", ""))
    ).lower()

    if "gpt2" in model_type:
        num_layers = int(text_config.n_layer)
        num_heads = int(text_config.n_head)
        head_dim = int(getattr(text_config, "head_dim", text_config.n_embd // num_heads))
        max_positions = int(text_config.n_positions)
        eligible_layers = tuple(range(num_layers))
    elif "qwen" in model_type or "llama" in model_type:
        num_layers = int(text_config.num_hidden_layers)
        num_heads = int(text_config.num_attention_heads)
        hidden_size = int(text_config.hidden_size)
        head_dim = int(getattr(text_config, "head_dim", hidden_size // num_heads))
        max_positions = int(getattr(text_config, "max_position_embeddings", 2048))
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            eligible_layers = tuple(range(num_layers))
        else:
            eligible_layers = tuple(
                index
                for index, layer_type in enumerate(layer_types)
                if layer_type == "full_attention"
            )
    else:
        raise ValueError(f"Unsupported model type for pruning: {model_type}")

    if not eligible_layers:
        raise ValueError(f"Model type {model_type} has no standard attention layers to prune")

    return ModelMetadata(
        model_type=model_type,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        eligible_layers=eligible_layers,
        max_position_embeddings=max_positions,
    )


def get_model_meta(model):
    """Backward-compatible tuple interface used by the original pruning scripts."""
    metadata = get_model_metadata(model)
    return (
        metadata.num_layers,
        metadata.num_heads,
        metadata.head_dim,
        metadata.model_type,
    )


def _find_gpt2_blocks(model):
    candidates = [model]
    seen: set[int] = set()
    while candidates:
        candidate = candidates.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        transformer = getattr(candidate, "transformer", None)
        if transformer is not None and hasattr(transformer, "h"):
            return transformer.h
        for attribute in ("base_model", "model"):
            child = getattr(candidate, attribute, None)
            if child is not candidate:
                candidates.append(child)
    raise AttributeError(f"Could not find GPT-2 transformer blocks in {type(model).__name__}")


def _find_decoder_layers(model):
    candidates = [model]
    seen: set[int] = set()
    while candidates:
        candidate = candidates.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        layers = getattr(candidate, "layers", None)
        if layers is not None and len(layers) > 0:
            layer = layers[0]
            if hasattr(layer, "self_attn") or hasattr(layer, "linear_attn"):
                return layers
        for attribute in ("base_model", "model", "language_model", "text_model"):
            child = getattr(candidate, attribute, None)
            if child is not candidate:
                candidates.append(child)
    raise AttributeError(f"Could not find decoder layers in {type(model).__name__}")


def get_output_projection(model, layer: int, model_type: str | None = None):
    metadata = get_model_metadata(model)
    resolved_type = (model_type or metadata.model_type).lower()

    if "gpt2" in resolved_type:
        return _find_gpt2_blocks(model)[layer].attn.c_proj

    layer_module = _find_decoder_layers(model)[layer]
    attention = getattr(layer_module, "self_attn", None)
    if attention is None:
        raise AttributeError(
            f"Layer {layer} is not a standard attention layer and cannot be head-pruned"
        )
    for attribute in ("o_proj", "out_proj"):
        projection = getattr(attention, attribute, None)
        if projection is not None:
            return projection
    raise AttributeError(
        f"Could not find an output projection for layer {layer} in {type(model).__name__}"
    )
