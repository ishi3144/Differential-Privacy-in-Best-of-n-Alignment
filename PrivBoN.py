import numpy as np

def priv_bon_gumbel(prompt, policy_model, reward_model, n, sigma):
    """
    Private Best-of-N (PrivBoN) using Gumbel noise.
    
    Args:
        prompt (str): The input prompt or question.
        policy_model (object): Object with a `.generate(prompt, num_samples)` method.
        reward_model (object): Object with a `.score(prompt, candidates)` method.
        n (int): Number of candidate responses to generate.
        sigma (float): The scale parameter for the Gumbel noise (> 0).
        
    Returns:
        str: The selected response.
    """
    # 1. Generate n samples from the base policy
    candidates = policy_model.generate(prompt, num_samples=n)
    
    # 2. Score all candidates using the reward model
    rewards = np.array(reward_model.score(prompt, candidates))
    
    # 3. Add independent Gumbel noise to the rewards
    # Gumbel(loc=0, scale=sigma)
    gumbel_noise = np.random.gumbel(loc=0.0, scale=sigma, size=n)
    noisy_scores = rewards + gumbel_noise
    
    # 4. Return the candidate with the maximum noisy score
    best_idx = np.argmax(noisy_scores)
    
    return candidates[best_idx]
