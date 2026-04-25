import numpy as np

def best_of_n(prompt, policy_model, reward_model, n):
    """
    Standard Best-of-N (BoN) Alignment.
    
    Args:
        prompt (str): The input prompt or question.
        policy_model (object): Object with a `.generate(prompt, num_samples)` method.
        reward_model (object): Object with a `.score(prompt, candidates)` method.
        n (int): Number of candidate responses to generate.
        
    Returns:
        str: The selected response with the maximum reward.
    """
    # 1. Generate n samples from the base policy
    candidates = policy_model.generate(prompt, num_samples=n)
    
    # 2. Score all candidates using the reward model
    rewards = np.array(reward_model.score(prompt, candidates))
    
    # 3. Return the candidate with the maximum reward
    best_idx = np.argmax(rewards)
    
    return candidates[best_idx]
