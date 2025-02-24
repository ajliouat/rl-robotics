from models.policies.ppo import PPO

def test_policy():
    policy = PPO()
    assert policy is not None