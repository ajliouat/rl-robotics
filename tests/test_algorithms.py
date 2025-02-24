from src.algorithms.ppo import PPO

def test_ppo():
    ppo = PPO()
    assert ppo is not None