from src.envs.robotics_env import RoboticsEnv

def test_env():
    env = RoboticsEnv()
    obs = env.reset()
    assert obs is not None