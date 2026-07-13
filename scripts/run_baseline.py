import argparse
import time

from redundancy.data import get_wikitext_dataset
from redundancy.eval import evaluate_perplexity
from redundancy.models import RedundancyModel
from datasets import load_dataset
from tqdm import tqdm

import torch
import json


def main():
    parser = argparse.ArgumentParser(description="Run Baseline Evaluation")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--dataset", type=str, default="wikitext-103-raw-v1", help="Dataset name")
    args = parser.parse_args()

    print(f"Initializing baseline evaluation for: {args.model}")
    redundancy_model = RedundancyModel(args.model)
    redundancy_model.model.eval()

    dataset = get_wikitext_dataset(subset=args.dataset)

    avg_nll, perplexity, n_tokens = evaluate_perplexity(
        model=redundancy_model.model,
        tokenizer=redundancy_model.tokenizer,
        dataset=dataset,
        device=redundancy_model.device,
    )

    results = {
        "model": args.model,
        "dataset": args.dataset,
        "evaluation_method": "sliding_window_stride_512",
        "baseline_loss": round(avg_nll, 4),
        "baseline_perplexity": round(perplexity, 4),
        "total_tokens_evaluated": n_tokens,
        "hardware_device": str(redundancy_model.device),
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    }

    output_file = f"configs/experiments/baseline_results_{args.model}_{args.dataset}.json"
    with open(output_file, "w") as f:
        json.dump(results, f)

    print(f"Baseline evaluation completed. Results saved to {output_file}")


if __name__ == "__main__":
    main()
