import numpy as np
import traci
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from rl_gym_environments import SUMOEnv

from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self):
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()

# ----------------------------------------------- RL ENVIRONMENT START -----------------------------------------------

env = SUMOEnv()
try:
    check_env(env)
    print(env.observation_space)
    print(env.observation_space.sample(), env.action_space.sample())

    # ----------------------------------------------- RL NOW WITH MODEL -----------------------------------------------

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-3)

    # Train the agent
    total_timesteps = 12000
    progress_bar_callback = ProgressBarCallback(total_timesteps)
    model.learn(total_timesteps, callback=[progress_bar_callback], log_interval=100)

    # Plot mean speeds
    if len(env.mean_speeds) > 0:
        plt.plot(env.mean_speeds)
        plt.xlabel("Step")
        plt.ylabel("Mean Speed")
        plt.title("Mean Speed over Training")
        plt.show()
    else:
        print("No mean speeds data to plot")

    # Save the model
    model.save("models/ppo_sumo_model")

    # Evaluate the agent
    mean_reward = 0
    n_episodes = 1
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        mean_reward += episode_reward

    mean_reward /= n_episodes
    print(f"Mean reward over {n_episodes} episodes: {mean_reward}")

finally:
    env.close()