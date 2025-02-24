import yaml
from src.envs.robotics_env import RoboticsEnv
from models.policies.ppo import PPO

def train(config):
    # Initialize environment
    env = RoboticsEnv()

    # Initialize PPO policy
    policy = PPO(env, config)

    # Train policy
    policy.train()

    # Save policy
    policy.save("models/checkpoints/ppo_policy.pth")