from datasets import load_dataset


def get_wikitext_dataset(
    split="test", subset="wikitext-103-raw-v1", cache_dir="./configs/datasets"
):
    print("Loading dataset")
    dataset = load_dataset("Salesforce/wikitext", subset, cache_dir=cache_dir, split=split)
    print(f"Loaded dataset with {len(dataset)} samples")
    return dataset
