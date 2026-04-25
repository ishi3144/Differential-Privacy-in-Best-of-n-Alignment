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
        # Valid if it falls before the next largest reward
        if k == n or lam >= sorted_r[k]:
            return lam
    return lam

def inference_time_pessimism(prompt, policy_model, reward_model, n, beta):
    """
    Standard Inference-Time Pessimism (ITP).
    
    Args:
        prompt (str): The input prompt or question.
        policy_model (object): Object with a `.generate(prompt, num_samples)` method.
        reward_model (object): Object with a `.score(prompt, candidates)` method.
        n (int): Number of candidate responses to generate.
        beta (float): The pessimism/regularization penalty parameter (> 0).
        
    Returns:
        str: The selected response via rejection sampling.
    """
    # 1. Generate n samples
    candidates = policy_model.generate(prompt, num_samples=n)
    rewards = np.array(reward_model.score(prompt, candidates))
    
    # 2. Compute normalization constant lambda
    lam_hat = find_lambda(rewards, beta)
    
    # 3. Calculate truncation parameter M
    r_max = np.max(rewards)
    M = max((r_max - lam_hat) / beta, 1e-8)
    
    # 4. Sequential rejection sampling
    for i in range(n):
        w_i = max((rewards[i] - lam_hat) / beta, 0.0)
        p_accept = min(w_i / M, 1.0)
        
        if np.random.rand() < p_accept:
            return candidates[i]
            
    # 5. Fallback to base policy if all candidates are rejected
    fallback_candidate = policy_model.generate(prompt, num_samples=1)[0]
    return fallback_candidate
