"""
utils.py
Shared utilities for inference-time alignment algorithms.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification


def load_base_policy(model_name: str, device: str = "cuda"):
    """Load a base policy (causal LM) and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    return model, tokenizer


def load_reward_model(rm_name: str, device: str = "cuda"):
    """Load a reward model and its tokenizer."""
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_name)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        rm_name, torch_dtype=torch.float16, device_map=device
    )
    rm_model.eval()
    return rm_model, rm_tokenizer


@torch.no_grad()
def generate_candidates(
    model, tokenizer, prompt: str, n: int,
    max_new_tokens: int = 256, temperature: float = 1.0,
    top_p: float = 1.0,
) -> List[str]:
    """Sample n candidate responses from the base policy."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=n,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    responses = [
        tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
        for o in outputs
    ]
    return responses


@torch.no_grad()
def score_responses(
    rm_model, rm_tokenizer, prompt: str, responses: List[str],
    batch_size: int = 8,
) -> np.ndarray:
    """Score a list of responses; returns shape-(n,) array of rewards."""
    scores = []
    for i in range(0, len(responses), batch_size):
        batch = responses[i:i+batch_size]
        texts = [prompt + r for r in batch]
        enc = rm_tokenizer(
            texts, padding=True, truncation=True,
            max_length=2048, return_tensors="pt"
        ).to(rm_model.device)
        logits = rm_model(**enc).logits.squeeze(-1)
        scores.extend(logits.float().cpu().numpy().tolist())
    return np.array(scores, dtype=np.float64)
