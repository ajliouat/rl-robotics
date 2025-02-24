import numpy as np

def calculate_success_rate(episode_rewards):
    return np.mean(episode_rewards > 0)