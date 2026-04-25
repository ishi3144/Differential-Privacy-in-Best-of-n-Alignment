"""
bon.py
Standard Best-of-N (BoN) inference-time alignment.

Algorithm:
  1. Sample n responses from the base policy.
  2. Score each response with the reward model.
  3. Return the response with the highest reward.
"""

import argparse
import numpy as np
from typing import List, Tuple

from utils import (
    load_base_policy, load_reward_model,
    generate_candidates, score_responses,
)


def best_of_n(
    prompt: str,
    base_model, base_tokenizer,
    reward_model, reward_tokenizer,
    n: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Run Best-of-N selection on a single prompt.

    Returns
    -------
    selected : str
        The chosen response.
    probs : np.ndarray, shape (n,)
        Selection probabilities (one-hot at argmax).
    rewards : np.ndarray, shape (n,)
        Reward scores assigned to each candidate.
    """
    candidates = generate_candidates(
        base_model, base_tokenizer, prompt, n,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    rewards = score_responses(reward_model, reward_tokenizer, prompt, candidates)

    i_star = int(np.argmax(rewards))
    probs = np.zeros(n)
    probs[i_star] = 1.0

    return candidates[i_star], probs, rewards


def main():
    p = argparse.ArgumentParser(description="Standard Best-of-N alignment.")
    p.add_argument("--base_model",  required=True,
                   help="HuggingFace ID or path of the base policy.")
    p.add_argument("--reward_model", required=True,
                   help="HuggingFace ID or path of the reward model.")
    p.add_argument("--prompt",      required=True,
                   help="Input prompt string.")
    p.add_argument("--n",            type=int, default=64,
                   help="Number of candidates to sample.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature",  type=float, default=1.0)
    p.add_argument("--device",       default="cuda")
    args = p.parse_args()

    print(f"Loading base model: {args.base_model}")
    base_model, base_tok = load_base_policy(args.base_model, args.device)
    print(f"Loading reward model: {args.reward_model}")
    rm_model, rm_tok = load_reward_model(args.reward_model, args.device)

    print(f"\nRunning BoN with N={args.n}...")
    selected, probs, rewards = best_of_n(
        args.prompt, base_model, base_tok, rm_model, rm_tok, args.n,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    )

    print(f"\nReward statistics: min={rewards.min():.4f}, "
          f"max={rewards.max():.4f}, mean={rewards.mean():.4f}")
    print(f"Selected (idx={int(np.argmax(probs))}):\n{selected}")


if __name__ == "__main__":
    main()
