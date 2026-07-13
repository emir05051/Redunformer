import random
import torch

from redundancy.hooks import HeadPruningHook


def get_model_meta(model):
    config = model.config
    model_type = getattr(config, "model_type", "").lower()

    if "gpt2" in model_type:
        # https://huggingface.co/docs/transformers/v5.13.1/en/model_doc/gpt2#transformers.GPT2Config
        num_layers = config.n_layer
        num_heads = config.n_head
        hidden_size = config.n_embd
    elif model_type in ["qwen", "llama"]:
        # https://huggingface.co/docs/transformers/v5.13.1/en/model_doc/qwen3#transformers.Qwen3Config
        # https://huggingface.co/docs/transformers/v5.13.1/en/model_doc/llama#transformers.LlamaConfig
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size
    else:
        raise ValueError(f"Unsupported model type for universal pruning: {model_type}")

    head_dim = hidden_size // num_heads
    return num_layers, num_heads, head_dim, model_type


def get_output_projection(model, layer, model_type):
    if model_type == "gpt2":
        return model.transformer.h[layer].attn.c_proj
    else:
        return model.model.layers[layer].self_attn.o_proj


def random_prune_model(model, sparsity: float, seed: int = 42):
    random.seed(seed)

    num_layers, num_heads, head_dim, model_type = get_model_meta(model)

    k = max(1, int(num_heads * sparsity)) if sparsity > 0 else 0
    if k == 0:
        print(f"No heads to prune for sparsity {sparsity}. Returning without pruning.")
        return []

    hooks = []

    for layer_idx in range(num_layers):
        heads_to_mask = random.sample(range(num_heads), k)

        proj_layer = get_output_projection(model, layer_idx, model_type)

        hook = proj_layer.register_forward_pre_hook(HeadPruningHook(heads_to_mask, head_dim))
        hooks.append(hook)

    actual_sparsity = (k / num_heads) * 100

    print(
        f"Successfully masked {k}/{num_heads} heads ({actual_sparsity:.1f}%) per layer in {model_type}."
    )
    return hooks


def prune_model(model, sparsity, seed=42):
    return random_prune_model(
        model=model,
        sparsity=sparsity,
        seed=seed,
    )
