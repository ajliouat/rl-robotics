import torch
from models.policies.ppo import PPO

def export_policy():
    policy = PPO()
    torch.save(policy.state_dict(), "models/checkpoints/ppo_policy.pth")