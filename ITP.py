"""
itp.py
InferenceTimePessimism (ITP) of Huang et al. (2025).

Algorithm:
  1. Sample n responses from the base policy.
  2. Score each response.
  3. Solve for lambda such that
        (1/n) sum_i relu( (r_i - lambda) / beta ) = 1
     via a sort-and-bucket dynamic-programming routine.
  4. Run rejection sampling with weights w_i = relu((r_i - lambda)/beta)
     and truncation M = max_i w_i.
  5. If no acceptance, fall back to a uniform draw from candidates.
"""

import argparse
import numpy as np
from typing import List, Tuple

from utils import (
    load_base_policy, load_reward_model,
    generate_candidates, score_responses,
)


def compute_norm_constant(rewards: np.ndarray, beta: float) -> float:
    """
    Find lambda s.t. (1/N) sum_i relu((r_i - lambda)/beta) = 1.
    O(N log N) sort-and-iterate.
    """
    N = len(rewards)
    sorted_r = np.sort(rewards)
    r_prev = -np.inf
    J = float(np.sum(sorted_r) / N)
    Z = 1.0
    lam = J - beta
    for i in range(N):
        lam = (J - beta) / Z
        r_curr = float(sorted_r[i])
        if (r_prev <= lam < r_curr) or (i == N - 1):
            return lam
        J -= r_curr / N
        Z -= 1.0 / N
        r_prev = r_curr
    return lam


def itp(
    prompt: str,
    base_model, base_tokenizer,
    reward_model, reward_tokenizer,
    n: int,
    beta: float,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    seed: int = None,
) -> Tuple[str, np.ndarray, np.ndarray, float]:
    """
    Run InferenceTimePessimism on a single prompt.

    Parameters
    ----------
    beta : float
        chi-squared regularization strength.

    Returns
    -------
    selected : str
    probs    : np.ndarray (n,)   marginal selection probabilities
    rewards  : np.ndarray (n,)
    lam_hat  : float             computed normalization constant
    """
    rng = np.random.default_rng(seed)

    candidates = generate_candidates(
        base_model, base_tokenizer, prompt, n,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    rewards = score_responses(reward_model, reward_tokenizer, prompt, candidates)

    lam_hat = compute_norm_constant(rewards, beta)
    w = np.maximum((rewards - lam_hat) / beta, 0.0)
    M_trunc = max(w.max(), 1e-12)
    p_accept = np.minimum(w / M_trunc, 1.0)

    # Sequential rejection sampling
    probs = np.zeros(n)
    not_accepted_yet = 1.0
    selected_idx = None
    for i in range(n):
        probs[i] = not_accepted_yet * p_accept[i]
        u = rng.uniform()
        if selected_idx is None and u < p_accept[i]:
            selected_idx = i
            break
        not_accepted_yet *= (1.0 - p_accept[i])

    if selected_idx is None:
        # Fallback: uniform draw from candidates
        selected_idx = rng.integers(0, n)

    return candidates[selected_idx], probs, rewards, lam_hat


def main():
    p = argparse.ArgumentParser(description="InferenceTimePessimism (ITP).")
    p.add_argument("--base_model",   required=True)
    p.add_argument("--reward_model", required=True)
    p.add_argument("--prompt",       required=True)
    p.add_argument("--n",            type=int,   default=64)
    p.add_argument("--beta",         type=float, required=True,
                   help="chi-squared regularization strength.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature",  type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=None)
    p.add_argument("--device",       default="cuda")
    args = p.parse_args()

    print(f"Loading base model: {args.base_model}")
    base_model, base_tok = load_base_policy(args.base_model, args.device)
    print(f"Loading reward model: {args.reward_model}")
    rm_model, rm_tok = load_reward_model(args.reward_model, args.device)

    print(f"\nRunning ITP with N={args.n}, beta={args.beta}...")
    selected, probs, rewards, lam_hat = itp(
        args.prompt, base_model, base_tok, rm_model, rm_tok,
        args.n, args.beta,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, seed=args.seed,
    )

    print(f"\nReward stats:    min={rewards.min():.4f}, "
          f"max={rewards.max():.4f}, mean={rewards.mean():.4f}")
    print(f"lambda_hat:      {lam_hat:.4f}")
    print(f"Sum of probs:    {probs.sum():.4f}  (should be ≤ 1; remainder = fallback)")
    print(f"Selected response:\n{selected}")


if __name__ == "__main__":
    main()
