import numpy as np

def find_lambda(rewards, beta):
    """
    Finds the normalization constant lambda such that:
    (1/n) * sum( max(0, (r_i - lambda)/beta) ) = 1
    """
    n = len(rewards)
    sorted_r = np.sort(rewards)[::-1]  # Sort descending
    sum_r = 0.0
    for k in range(1, n + 1):
        sum_r += sorted_r[k - 1]
        lam = (sum_r - n * beta) / k
        if k == n or lam >= sorted_r[k]:
            return lam
    return lam

def private_itp(prompt, policy_model, reward_model, n, beta, sigma_x, sigma_z, L=4.0):
    """
    Private Inference-Time Pessimism (PrivITP).
    
    Args:
        prompt (str): The input prompt or question.
        policy_model (object): Object with a `.generate(prompt, num_samples)` method.
        reward_model (object): Object with a `.score(prompt, candidates)` method.
        n (int): Number of candidate responses to generate per phase.
        beta (float): The pessimism/regularization penalty parameter (> 0).
        sigma_x (float): Noise scale for the normalization constant.
        sigma_z (float): Noise scale for the reward evaluation.
        L (float): Truncation slack parameter (standard deviation bound, default 4.0).
        
    Returns:
        str: The privately selected response.
    """
    # --- PHASE 1: Private Normalization ---
    # Draw n independent samples to calculate the private threshold
    phase1_candidates = policy_model.generate(prompt, num_samples=n)
    phase1_rewards = np.array(reward_model.score(prompt, phase1_candidates))
    
    # Calculate the exact lambda, then mask it with Gaussian noise
    lam_clean = find_lambda(phase1_rewards, beta)
    lam_tilde = lam_clean + np.random.normal(0, sigma_x)
    
    # Calculate Truncation bound (M)
    r_max = np.max(phase1_rewards)
    M = max((r_max + sigma_z * L - lam_tilde) / beta, 1e-8)
    
    # --- PHASE 2: Private Rejection Sampling ---
    # Draw n FRESH independent samples to evaluate
    phase2_candidates = policy_model.generate(prompt, num_samples=n)
    phase2_rewards_clean = np.array(reward_model.score(prompt, phase2_candidates))
    
    for i in range(n):
        # Apply independent Gaussian noise to the reward
        r_tilde_i = phase2_rewards_clean[i] + np.random.normal(0, sigma_z)
        
        # Calculate acceptance weight
        w_i = max((r_tilde_i - lam_tilde) / beta, 0.0)
        p_accept = min(w_i / M, 1.0)
        
        # Accept or reject
        if np.random.rand() < p_accept:
            return phase2_candidates[i]
            
    # --- FALLBACK ---
    # If all items are rejected, fallback to a fresh base policy sample
    fallback_candidate = policy_model.generate(prompt, num_samples=1)[0]
    return fallback_candidate
