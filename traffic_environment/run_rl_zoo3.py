import sys
from traffic_environment import register_custom_envs
from rl_zoo3.train import train
from rl_learn import base_sumo_port
from traffic_environment.rl_gym_environments import num_envs_per_model
import argparse
ports = []

def setup(model):
    for i in range(num_envs_per_model):
        ports.append(base_sumo_port + i)

    for idx, port in enumerate(ports):
        register_custom_envs(port=port, model_name=model, model_idx=idx, day_idx=0)
    
def check_setup(model):
    try:
        import rl_zoo3
    except ModuleNotFoundError:
        print("rl_zoo3 is not accessible")

    import gymnasium as gym
    env_specs = gym.envs.registry.values()
    env_ids = [spec.id for spec in env_specs]
    for idx, port in enumerate(ports):
        assert f'TrafficEnv-{model}-{idx}-v0' in env_ids, "Environment not registered!"

# if __name__ == "__main__":
    model = 'DQN'
    ports = []
    setup(model)
    check_setup(model)

    # Ensure that the TrafficEnv-v0 params are added in the specific model
    # D:\phd_ws\speed_harmo\phd_speed_harmo_v3\.py310_tf_env\lib\site-packages\rl_zoo3\hyperparams\dqn.yml

    envs = []
    for idx, port in enumerate(ports):
        envs.append(f"TrafficEnv-{model}-{idx}-v0")
   
    sys.argv = [
        'train.py',  # Placeholder for the script name
        '--algo', 'dqn',
        # '--eval-freq', '10000',
        # '--save-freq', '25000',
        # '--n-timesteps', '300000',
        '--optimize',
        '--n-trials', '100',
        '--n-jobs', '2',
        '--env',
    ]

    # Iterate over each environment and execute the training script separately
    # for env in envs:
    #     sys.argv.append(env)
    sys.argv.append(envs[0])

    sys.argv.append('--tensorboard-log')
    sys.argv.append('./logs/')

    # Call train() as if it were run from the command line
    train()