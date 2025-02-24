import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, action_space.shape[0])
        self.critic = nn.Sequential(
            nn.Linear(obs_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 1))

    def forward(self, x):
        return self.actor(x), self.critic(x)