import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_checker import check_env

from rl_gym_environments import SUMOEnv

env = SUMOEnv()

model_paths = {
    "PPO": "rl_models/PPO/best_model",
    "A2C": "rl_models/A2C/best_model",
    "DQN": "rl_models/DQN/best_model"
}

def run_agent(env, model):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
    
    return {
        'mean_speeds': env.mean_speeds,
        'flows': env.flows,
        'emissions_over_time': env.emissions_over_time,
        'cvs_seg_time': env.cvs_seg_time
    }

def save_metrics(metrics, agent_name):
    with open(f'./metrics/{agent_name}_metrics.csv', 'w+') as metrics_file:
        list_to_string = lambda x: ','.join([str(elem) for elem in x]) + '\n'
        metrics_file.write(list_to_string(metrics['mean_speeds']))
        metrics_file.write(list_to_string(metrics['flows']))
        metrics_file.write(list_to_string(metrics['emissions_over_time']))
    
    pd.DataFrame(metrics['cvs_seg_time']).to_csv(f'./metrics/{agent_name}.csv', index=False, header=False)

try:
    check_env(env)
    
    results = {}
    for agent_name, model_path in model_paths.items():
        if agent_name == "PPO":
            model = PPO.load(model_path)
        elif agent_name == "A2C":
            model = A2C.load(model_path)
        elif agent_name == "DQN":
            model = DQN.load(model_path)
        
        results[agent_name] = run_agent(env, model)
        save_metrics(results[agent_name], agent_name)
    
    plt.figure(figsize=(15, 5))

    # Flow
    plt.subplot(131)
    for agent_name, data in results.items():
        plt.plot(data['flows'], label=agent_name)
    plt.xlabel("Iteration")
    plt.ylabel("Flow")
    plt.title("Flow Comparison")
    plt.legend()

    # Mean speed
    plt.subplot(132)
    for agent_name, data in results.items():
        plt.plot(data['mean_speeds'], label=agent_name)
    plt.xlabel("Iteration")
    plt.ylabel("Mean speed")
    plt.title("Mean Speed Comparison")
    plt.legend()

    # Emissions
    plt.subplot(133)
    for agent_name, data in results.items():
        plt.plot(data['emissions_over_time'], label=agent_name)
    plt.xlabel("Step")
    plt.ylabel("Emission level")
    plt.title("Emissions Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()

finally:
    env.close()