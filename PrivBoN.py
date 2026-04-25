"""
privbon_gumbel.py
Private Best-of-N via Gumbel-Max (Exponential Mechanism).

Algorithm (PrivBoN):
  1. Sample n responses from the base policy.
  2. Compute reward r_i for each, then add Gumbel(0, sigma) noise:
        tilde_r_i = r_i + g_i,   g_i ~ Gumbel(0, sigma)
  3. Return argmax_i tilde_r_i.

By the Gumbel-Max trick, this is equivalent to sampling
   P(i) propto exp(r_i / sigma).

DP guarantee: (eps, 0)-DP with sigma = 2 * Delta_inf / eps,
where Delta_inf is the per-query L_inf sensitivity of r_hat.
"""

import argparse
import numpy as np
from typing import List, Tuple

from utils import (
    load_base_policy, load_reward_model,
    generate_candidates, score_responses,
)


def sample_gumbel(size, scale=1.0, rng=None):
    """Draw Gumbel(0, scale) noise. Uses inverse-CDF sampling."""
    rng = rng if rng is not None else np.random
    u = rng.uniform(low=1e-12, high=1.0 - 1e-12, size=size)
    return -scale * np.log(-np.log(u))


def priv_bon_gumbel(
    prompt: str,
    base_model, base_tokenizer,
    reward_model, reward_tokenizer,
    n: int,
    sigma: float,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    seed: int = None,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """
    PrivBoN with Gumbel noise (Exponential Mechanism).

    Parameters
    ----------
    sigma : float
        Gumbel noise scale (in raw reward units).
        For (eps, 0)-DP: sigma = 2 * Delta_inf / eps.

    Returns
    -------
    selected   : str               selected response
    probs      : np.ndarray (n,)   one-hot at chosen index
    rewards    : np.ndarray (n,)   clean rewards
    noisy_r    : np.ndarray (n,)   noisy rewards used for selection
    """
    rng = np.random.default_rng(seed)

    candidates = generate_candidates(
        base_model, base_tokenizer, prompt, n,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    rewards = score_responses(reward_model, reward_tokenizer, prompt, candidates)

    if sigma <= 0:
        i_star = int(np.argmax(rewards))
        noisy_r = rewards.copy()
    else:
        g = sample_gumbel(size=n, scale=sigma, rng=rng)
        noisy_r = rewards + g
        i_star = int(np.argmax(noisy_r))

    probs = np.zeros(n)
    probs[i_star] = 1.0

    return candidates[i_star], probs, rewards, noisy_r


def main():
    p = argparse.ArgumentParser(description="PrivBoN with Gumbel noise.")
    p.add_argument("--base_model",   required=True)
    p.add_argument("--reward_model", required=True)
    p.add_argument("--prompt",       required=True)
    p.add_argument("--n",            type=int,   default=64)
    p.add_argument("--sigma",        type=float, required=True,
                   help="Gumbel noise scale (raw reward units).")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature",  type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=None)
    p.add_argument("--device",       default="cuda")
    args = p.parse_args()

    print(f"Loading base model: {args.base_model}")
    base_model, base_tok = load_base_policy(args.base_model, args.device)
    print(f"Loading reward model: {args.reward_model}")
    rm_model, rm_tok = load_reward_model(args.reward_model, args.device)

    print(f"\nRunning PrivBoN-Gumbel with N={args.n}, sigma={args.sigma}...")
    selected, probs, rewards, noisy_r = priv_bon_gumbel(
        args.prompt, base_model, base_tok, rm_model, rm_tok,
        args.n, args.sigma,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, seed=args.seed,
    )

    print(f"\nClean reward stats:  min={rewards.min():.4f}, "
          f"max={rewards.max():.4f}, mean={rewards.mean():.4f}")
    print(f"Noisy reward stats:  min={noisy_r.min():.4f}, "
          f"max={noisy_r.max():.4f}, mean={noisy_r.mean():.4f}")
    print(f"Selected (idx={int(np.argmax(probs))}, "
          f"clean reward={rewards[int(np.argmax(probs))]:.4f}):\n{selected}")


if __name__ == "__main__":
    main()
