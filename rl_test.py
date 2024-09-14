import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from rl_gym_environments import SUMOEnv

env = SUMOEnv()

try:
    check_env(env)

    # Load the trained model
    model = PPO.load("models/ppo_sumo_model")

    # Test the model
    obs, _ = env.reset()
    done = False

    while not done:
        action, state = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

    # Save metrics into csv file
    approach = 'rl_ppo'
    with open(f'./metrics/{approach}_metrics.csv', 'w+') as metrics_file:
        list_to_string = lambda x: ','.join([str(elem) for elem in x]) + '\n'
        metrics_file.write(list_to_string(env.mean_speeds))
        metrics_file.write(list_to_string(env.flows))
        metrics_file.write(list_to_string(env.emissions_over_time))

    pd.DataFrame(env.cvs_seg_time).to_csv(f'./metrics/{approach}_cvs.csv', index=False, header=False)

    # Plot results
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.xlabel("Iteration")
    plt.ylabel("Flow")
    plt.title("Flow")
    plt.plot(env.flows)

    plt.subplot(132)
    plt.xlabel("Iteration")
    plt.ylabel("Mean speed")
    plt.title("Mean speed over the whole stretch")
    plt.plot(env.mean_speeds)

    plt.subplot(133)
    plt.xlabel("Step")
    plt.ylabel("Emission level")
    plt.title("Emissions over time")
    plt.plot(env.emissions_over_time)

    plt.tight_layout()
    plt.show()
finally:
    env.close()