import argparse
import json
import time

from redundancy.data import get_wikitext_dataset
from redundancy.eval import evaluate_perplexity
from redundancy.models import RedundancyModel
from redundancy.pruning.random_pruning import prune_model


def main():
    print("Starting pruning evaluation script...")
    parser = argparse.ArgumentParser(description="Run Pruning Evaluation")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--dataset", type=str, default="wikitext-103-raw-v1", help="Dataset name")
    parser.add_argument(
        "--ratio", type=float, default=0.2, help="Percentage of heads to prune (0.0 to 1.0)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for head selection")
    args = parser.parse_args()

    print(f"Initializing pruning evaluation for: {args.model} at {args.ratio*100}% sparsity")
    redundancy_model = RedundancyModel(args.model)
    redundancy_model.model.eval()

    active_hooks = prune_model(model=redundancy_model.model, sparsity=args.ratio, seed=args.seed)

    dataset = get_wikitext_dataset(subset=args.dataset)

    avg_nll, perplexity, n_tokens = evaluate_perplexity(
        model=redundancy_model.model,
        tokenizer=redundancy_model.tokenizer,
        dataset=dataset,
        device=redundancy_model.device,
    )

    results = {
        "model": args.model,
        "pruning_method": "random_head_pruning",
        "pruning_ratio": args.ratio,
        "random_seed": args.seed,
        "dataset": args.dataset,
        "pruned_loss": round(avg_nll, 4),
        "pruned_perplexity": round(perplexity, 4),
        "total_tokens_evaluated": n_tokens,
        "hardware_device": str(redundancy_model.device),
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    }

    for hook in active_hooks:
        hook.remove()

    output_file = (
        f"configs/experiments/pruned_results_{args.model}_{args.dataset}_{args.ratio}.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f)

    print(f"Pruning evaluation completed. Results saved to {output_file}")


if __name__ == "__main__":
    main()
