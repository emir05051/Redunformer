import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from dotenv import load_dotenv

load_dotenv()


class RedundancyModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = Accelerator().device

        local_model_cache = "./configs/models"
        hf_token = os.getenv("HF_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=local_model_cache, token=hf_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=local_model_cache, token=hf_token
        )
        self.model.to(self.device)
