from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator


class RedundancyModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = Accelerator().device

        local_model_cache = "./configs/models"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_model_cache)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_model_cache)
        self.model.to(self.device)
