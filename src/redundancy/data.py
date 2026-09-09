from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


def get_wikitext_dataset(
    split="test", subset="wikitext-103-raw-v1", cache_dir="./configs/datasets"
):
    print("Loading WikiText dataset")
    dataset = load_dataset("Salesforce/wikitext", subset, cache_dir=cache_dir, split=split)
    print(f"Loaded WikiText with {len(dataset)} samples")
    return dataset


@dataclass(frozen=True)
class C4SubsetMetadata:
    dataset: str
    configuration: str
    split: str
    revision: str
    seed: int
    shuffle_buffer_size: int
    requested_tokens: int
    actual_tokens: int
    sequence_length: int
    num_sequences: int
    tokenizer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PackedTokenDataset(Dataset):
    def __init__(self, blocks: list[list[int]], metadata: C4SubsetMetadata):
        self._blocks = [torch.tensor(block, dtype=torch.long) for block in blocks]
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids = self._blocks[index]
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


def _resolved_dataset_revision(dataset_name: str, revision: str) -> str:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(dataset_name, revision=revision).sha
    except Exception as error:
        print(f"Warning: could not resolve C4 revision {revision!r}: {error}")
        return revision


def pack_text_stream(
    documents: Iterable[dict[str, Any]],
    tokenizer,
    max_train_tokens: int,
    sequence_length: int,
) -> list[list[int]]:
    if max_train_tokens < sequence_length:
        raise ValueError("max_train_tokens must be at least one sequence")
    if sequence_length <= 1:
        raise ValueError("sequence_length must be greater than one")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id for document separation")

    target_blocks = max_train_tokens // sequence_length
    blocks: list[list[int]] = []
    pending: list[int] = []
    for document in documents:
        text = document.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        pending.extend(token_ids)
        pending.append(eos_token_id)
        while len(pending) >= sequence_length and len(blocks) < target_blocks:
            blocks.append(pending[:sequence_length])
            del pending[:sequence_length]
        if len(blocks) == target_blocks:
            break

    if len(blocks) != target_blocks:
        actual_tokens = len(blocks) * sequence_length
        raise ValueError(
            f"C4 stream ended after {actual_tokens} packed tokens; "
            f"needed {target_blocks * sequence_length}"
        )
    return blocks


def get_c4_recovery_dataset(
    tokenizer,
    max_train_tokens: int = 1_000_000,
    sequence_length: int = 512,
    seed: int = 42,
    shuffle_buffer_size: int = 10_000,
    revision: str = "main",
    cache_dir: str = "./configs/datasets",
    source_dataset=None,
) -> PackedTokenDataset:
    dataset_name = "allenai/c4"
    configuration = "en"
    split = "train"
    resolved_revision = revision
    if source_dataset is None:
        resolved_revision = _resolved_dataset_revision(dataset_name, revision)
        source_dataset = load_dataset(
            dataset_name,
            configuration,
            split=split,
            revision=resolved_revision,
            cache_dir=cache_dir,
            streaming=True,
        )
        source_dataset = source_dataset.shuffle(
            seed=seed, buffer_size=shuffle_buffer_size
        )

    blocks = pack_text_stream(
        documents=source_dataset,
        tokenizer=tokenizer,
        max_train_tokens=max_train_tokens,
        sequence_length=sequence_length,
    )
    metadata = C4SubsetMetadata(
        dataset=dataset_name,
        configuration=configuration,
        split=split,
        revision=resolved_revision,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size,
        requested_tokens=max_train_tokens,
        actual_tokens=len(blocks) * sequence_length,
        sequence_length=sequence_length,
        num_sequences=len(blocks),
        tokenizer=getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
    )
    return PackedTokenDataset(blocks, metadata)
