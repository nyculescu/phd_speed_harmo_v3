from .rl_gym_environments import TrafficEnv
import gymnasium as gym
from gymnasium.envs.registration import register

def make_traffic_env(port, model, model_idx, day_idx):
    def _init():
        trafficEnv = TrafficEnv(port=port, model=model, model_idx=model_idx)
        trafficEnv.day_index = (day_idx + model_idx) % 7
        return trafficEnv
    return _init

def register_custom_envs(port, model_name, model_idx, day_idx):
    register(
        id=f'TrafficEnv-{model_name}-{model_idx}-v0',
        entry_point=lambda: make_traffic_env(port, model_name, model_idx, day_idx)(),
    )
    
    # test_env = gym.make('TrafficEnv-v0')
    # print("Environment created successfully!")
