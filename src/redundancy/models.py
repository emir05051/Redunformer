from __future__ import annotations

import os

import torch
from accelerate import Accelerator
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()


def should_quantize_model(model_name: str, quantization: str = "auto") -> bool:
    if quantization not in {"auto", "4bit", "none"}:
        raise ValueError("quantization must be one of: auto, 4bit, none")
    if quantization == "4bit":
        return True
    if quantization == "none":
        return False
    normalized = model_name.lower()
    return "qwen" in normalized or "llama" in normalized


class RedundancyModel:
    def __init__(
        self,
        model_name: str,
        quantization: str = "none",
        revision: str | None = None,
    ):
        self.model_name = model_name
        self.revision = revision
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.is_quantized = should_quantize_model(model_name, quantization)

        local_model_cache = "./configs/models"
        hf_token = os.getenv("HF_TOKEN")
        common = {
            "cache_dir": local_model_cache,
            "token": hf_token,
            "revision": revision,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **common)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_arguments = dict(common)
        model_arguments["dtype"] = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        if self.is_quantized:
            if self.device.type != "cuda":
                raise RuntimeError("4-bit QLoRA loading requires a CUDA device")
            model_arguments["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_arguments["device_map"] = {"": self.accelerator.local_process_index}

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_arguments)
        if not self.is_quantized:
            self.model.to(self.device)
