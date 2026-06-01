from redundancy.models import RedundancyModel
from datasets import load_dataset
from tqdm import tqdm

import torch
import json

# Source: https://huggingface.co/docs/transformers/perplexity#example-calculating-perplexity-with-gpt-2-in-transformers


def run_baseline():
    print("Start baseline evaluation")

    print("Initializing model")
    model_name = "gpt2"
    redundancy_model = RedundancyModel(model_name)
    print(f"Loaded model: {redundancy_model.model_name}")

    redundancy_model.model.eval()

    # Use wikitext-2 dataset suggested in the pdf for evaluation
    # https://huggingface.co/datasets/Salesforce/wikitext has other versions of wikitext
    print("Loading dataset")
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir="./configs/datasets", split="test"
    )
    print(f"Loaded dataset with {len(dataset)} samples")

    encodings = redundancy_model.tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    max_length = redundancy_model.model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(redundancy_model.device)
        target_ids = input_ids.clone()

        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = redundancy_model.model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        num_valid_tokens = (target_ids != -100).sum().item()
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size

        nll_sum += neg_log_likelihood.item() * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens
    perplexity = torch.exp(torch.tensor(avg_nll)).item()

    results = {
        "model": model_name,
        "dataset": "wikitext-2-raw-v1 (test split)",
        "evaluation_method": "sliding_window_stride_512",
        "baseline_loss": round(avg_nll, 4),
        "baseline_perplexity": round(perplexity, 4),
        "total_tokens_evaluated": n_tokens,
        "hardware_device": str(redundancy_model.device),
    }

    output_file = "configs/experiments/baseline_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f)

    print(f"Baseline evaluation completed. Results saved to {output_file}")


if __name__ == "__main__":
    run_baseline()
