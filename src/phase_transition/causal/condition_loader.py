"""Shared model loading for fixed-response causal experiments."""

import gc

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(base_model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_condition_model(
    base_model_path: str,
    adapter_dir: str,
    dtype: torch.dtype,
    attn_implementation: str,
):
    kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": attn_implementation,
    }
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    return model


def release_model(model=None) -> None:
    """Collect unused models; callers must first drop their own model references."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
