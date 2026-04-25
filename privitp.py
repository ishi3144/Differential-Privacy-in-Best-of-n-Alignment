"""
privitp.py
Private InferenceTimePessimism (PrivITP) — two-phase Gaussian DP.

Algorithm (Algorithm 1 in the paper):

  Phase 1 (private normalization):
    1. Draw y_1, ..., y_n ~ pi_0(.|x); compute r_i = r_hat(x, y_i)
    2. Find lambda(x) s.t. (1/n) sum_i relu( (r_i - lambda) / beta ) = 1
    3. Set tilde_lambda = lambda + N(0, sigma_X^2)
       and M = (R_max + sigma_Z * L - tilde_lambda) / beta

  Phase 2 (private rejection sampling):
    4. Draw fresh y'_1, ..., y'_n ~ pi_0(.|x); compute
         tilde_r_i = r_hat(x, y'_i) + N(0, sigma_Z^2)
    5. For i = 1, ..., n:
         w_i = relu( (tilde_r_i - tilde_lambda) / beta )
         xi_i ~ Bernoulli( min(w_i / M, 1) )
         if xi_i == 1: return y'_i
    6. Fallback: return y'_{n+1} ~ pi_0(.|x)
"""

import argparse
import numpy as np
from typing import Tuple

from utils import (
    load_base_policy, load_reward_model,
    generate_candidates, score_responses,
)


def compute_norm_constant(rewards: np.ndarray, beta: float) -> float:
    """Same as in itp.py — kept here for self-containment."""
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


def priv_itp(
    prompt: str,
    base_model, base_tokenizer,
    reward_model, reward_tokenizer,
    n: int,
    beta: float,
    sigma_X: float,
    sigma_Z: float,
    L: float = 3.0,
    R_max: float = None,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    seed: int = None,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    PrivITP with two-phase Gaussian DP.

    Parameters
    ----------
    sigma_X : float
        Std of Gaussian noise added to lambda (Phase 1).
        Controls privacy of the normalization-constant release.
    sigma_Z : float
        Std of Gaussian noise added to rewards in Phase 2.
        Controls privacy of the rejection-sampling decisions.
    L : float
        Truncation parameter; M is conservatively set so that
        rejection sampling acceptance probabilities stay in [0,1]
        with high probability.
    R_max : float
        Upper bound on raw reward magnitudes. If None, taken as
        max of phase-1 rewards.

    Returns
    -------
    selected : str
    info     : dict   diagnostic info including lam, lam_tilde, M, fallback flag
    """
    rng = np.random.default_rng(seed)

    # ---- Phase 1: private normalization ----
    candidates_p1 = generate_candidates(
        base_model, base_tokenizer, prompt, n,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    rewards_p1 = score_responses(reward_model, reward_tokenizer, prompt, candidates_p1)

    lam = compute_norm_constant(rewards_p1, beta)
    g_lam = rng.normal(0.0, sigma_X) if sigma_X > 0 else 0.0
    lam_tilde = lam + g_lam

    if R_max is None:
        R_max = float(rewards_p1.max())

    M = max((R_max + sigma_Z * L - lam_tilde) / beta, 1e-12)

    # ---- Phase 2: private rejection sampling ----
    candidates_p2 = generate_candidates(
        base_model, base_tokenizer, prompt, n,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    rewards_p2 = score_responses(reward_model, reward_tokenizer, prompt, candidates_p2)

    g_r = rng.normal(0.0, sigma_Z, size=n) if sigma_Z > 0 else np.zeros(n)
    tilde_r = rewards_p2 + g_r

    w = np.maximum((tilde_r - lam_tilde) / beta, 0.0)
    p_accept = np.minimum(w / M, 1.0)

    selected_idx = None
    fallback = False
    for i in range(n):
        u = rng.uniform()
        if u < p_accept[i]:
            selected_idx = i
            break

    if selected_idx is None:
        # Fallback: draw a fresh sample from the base policy
        fallback = True
        fb = generate_candidates(
            base_model, base_tokenizer, prompt, 1,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        selected_response = fb[0]
        selected_score = score_responses(
            reward_model, reward_tokenizer, prompt, [selected_response]
        )[0]
    else:
        selected_response = candidates_p2[selected_idx]
        selected_score = float(rewards_p2[selected_idx])

    info = {
        "lam":         float(lam),
        "lam_tilde":   float(lam_tilde),
        "M":           float(M),
        "fallback":    fallback,
        "selected_idx": selected_idx if not fallback else -1,
        "selected_clean_reward": selected_score,
    }
    return selected_response, rewards_p1, rewards_p2, p_accept, info


def main():
    p = argparse.ArgumentParser(description="Private InferenceTimePessimism (PrivITP).")
    p.add_argument("--base_model",   required=True)
    p.add_argument("--reward_model", required=True)
    p.add_argument("--prompt",       required=True)
    p.add_argument("--n",            type=int,   default=64)
    p.add_argument("--beta",         type=float, required=True,
                   help="chi-squared regularization strength.")
    p.add_argument("--sigma_X",      type=float, required=True,
                   help="Gaussian noise std on lambda (Phase 1).")
    p.add_argument("--sigma_Z",      type=float, required=True,
                   help="Gaussian noise std on rewards (Phase 2).")
    p.add_argument("--L",            type=float, default=3.0,
                   help="Truncation parameter.")
    p.add_argument("--R_max",        type=float, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature",  type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=None)
    p.add_argument("--device",       default="cuda")
    args = p.parse_args()

    print(f"Loading base model: {args.base_model}")
    base_model, base_tok = load_base_policy(args.base_model, args.device)
    print(f"Loading reward model: {args.reward_model}")
    rm_model, rm_tok = load_reward_model(args.reward_model, args.device)

    print(f"\nRunning PrivITP with N={args.n}, beta={args.beta}, "
          f"sigma_X={args.sigma_X}, sigma_Z={args.sigma_Z}, L={args.L}...")

    selected, rewards_p1, rewards_p2, p_accept, info = priv_itp(
        args.prompt, base_model, base_tok, rm_model, rm_tok,
        n=args.n, beta=args.beta,
        sigma_X=args.sigma_X, sigma_Z=args.sigma_Z,
        L=args.L, R_max=args.R_max,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, seed=args.seed,
    )

    print(f"\nPhase 1 rewards: min={rewards_p1.min():.4f}, max={rewards_p1.max():.4f}")
    print(f"Phase 2 rewards: min={rewards_p2.min():.4f}, max={rewards_p2.max():.4f}")
    print(f"lambda:          {info['lam']:.4f}")
    print(f"lambda_tilde:    {info['lam_tilde']:.4f}  "
          f"(noise applied: {info['lam_tilde'] - info['lam']:.4f})")
    print(f"M (truncation):  {info['M']:.4f}")
    print(f"Fallback used:   {info['fallback']}")
    print(f"Selected idx:    {info['selected_idx']}, "
          f"clean reward: {info['selected_clean_reward']:.4f}")
    print(f"Selected response:\n{selected}")


if __name__ == "__main__":
    main()
