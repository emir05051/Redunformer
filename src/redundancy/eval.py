from __future__ import annotations

import torch
from tqdm import tqdm

from redundancy.utils import get_model_metadata


def evaluate_perplexity(
    model,
    tokenizer,
    dataset,
    device,
    stride: int = 512,
    window_size: int | None = None,
):
    texts = [text for text in dataset["text"] if text and text.strip()]
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    context_limit = get_model_metadata(model).max_position_embeddings
    max_length = min(window_size or 1024, context_limit)
    stride = min(stride, max_length)
    sequence_length = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    previous_end = 0
    was_training = model.training
    model.eval()
    try:
        for begin in tqdm(range(0, sequence_length, stride)):
            end = min(begin + max_length, sequence_length)
            target_length = end - previous_end
            input_ids = encodings.input_ids[:, begin:end].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-target_length] = -100

            with torch.no_grad():
                loss = model(input_ids=input_ids, labels=target_ids).loss

            valid_tokens = (target_ids != -100).sum().item()
            loss_tokens = valid_tokens - target_ids.size(0)
            if loss_tokens > 0:
                nll_sum += loss.item() * loss_tokens
                n_tokens += loss_tokens
            previous_end = end
            if end == sequence_length:
                break
    finally:
        model.train(was_training)

    if n_tokens == 0:
        raise ValueError("Evaluation dataset did not contain enough tokens")
    average_nll = nll_sum / n_tokens
    perplexity = torch.exp(torch.tensor(average_nll)).item()
    return average_nll, perplexity, n_tokens


def evaluate_lm_harness(model, tokenizer, device, tasks):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")
    results = lm_eval.simple_evaluate(model=lm_eval_model, tasks=tasks, log_samples=False)
    return results.get("results", {})
