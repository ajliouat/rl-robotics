import gym
import isaacgym

class RoboticsEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = isaacgym.make("RoboticsEnv")

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)