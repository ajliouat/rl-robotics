import yaml
from src.envs.robotics_env import RoboticsEnv
from models.policies.ppo import PPO

def evaluate(config):
    # Initialize environment
    env = RoboticsEnv()

    # Initialize PPO policy
    policy = PPO(env, config)

    # Load trained policy
    policy.load(config["policy"]["checkpoint"])

    # Evaluate policy
    obs = env.reset()
    done = False
    while not done:
        action, _ = policy(obs)
        obs, reward, done, info = env.step(action)