from matplotlib import pyplot as plt
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback

from rl_gym_environments import SUMOEnv

from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

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

env = SUMOEnv()

try:
    check_env(env)
    print(env.observation_space)
    print(env.observation_space.sample(), env.action_space.sample())

    # Define log directories
    logs_dirs = {
        "PPO": "./logs/PPO/",
        "A2C": "./logs/A2C/",
        "DQN": "./logs/DQN/"
    }
    models_dirs = {
        "PPO": "./rl_models/PPO/",
        "A2C": "./rl_models/A2C/",
        "DQN": "./rl_models/DQN/"
    }
    # Ensure directories exist
    for log_dir in logs_dirs.values():
        os.makedirs(log_dir, exist_ok=True)
    for model_dir in models_dirs.values():
        os.makedirs(model_dir, exist_ok=True)

    '''
    Factors to Consider to determine the appropriate number of total timesteps
    1. Environment Complexity: SUMOEnv simulates traffic scenarios with a discrete action space of 11 actions, representing speed limits. The observation space is a simple 1-dimensional Box, indicating a relatively straightforward state representation. This simplicity suggests that fewer timesteps might be needed compared to more complex environments.
    2. Simulation Length: Each episode in the environment runs for 120 steps (sim_length is set to 3600 divided by aggregation_time of 30). This means that the agent has 120 opportunities to learn from each episode.
    3. Desired Performance: If the goal is to achieve a basic level of performance or to test feasibility, starting with fewer timesteps is reasonable. High performance -> more timesteps
    4. Computational Resources: The availability of computational power can limit or extend the feasible number of timesteps. Training on GPUs can significantly speed up the process.
    '''
    total_timesteps = {
        #FIXME: current values just for testing
        "PPO": 100000, # 500k is generally robust and can handle fewer timesteps effectively
        "A2C": 60000, # 300k can learn efficiently with fewer samples due to its synchronous nature
        "DQN": 80000  # 400k is sample-efficient, especially in discrete action spaces
    }

    ''' Initialize models with recommended hyperparameters 
    Note about policies:
        - Use "MlpPolicy" for environments with continuous state spaces, as it provides a robust architecture for handling complex observations
        - Use "MlpPolicy" for vector inputs or "CnnPolicy" if dealing with image-based inputs
        Ref.: De La Fuente, N., & Guerra, D. A. V. (2024). A Comparative Study of Deep Reinforcement Learning Models: DQN vs PPO vs A2C. arXiv preprint arXiv:2407.14151. -> link: https://arxiv.org/html/2407.14151v1
    '''
    ppo_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4, # commonly used for stability and efficiency
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=logs_dirs["PPO"]
    )

    a2c_model = A2C(
        "MlpPolicy",
        env,
        learning_rate=7e-4, # typical due to its more aggressive updates
        n_steps=5,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=logs_dirs["A2C"]
    )

    dqn_model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4, # helps stabilize Q-value updates
        buffer_size=100000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=logs_dirs["DQN"]
    )
    ''' Learning rate values: https://arxiv.org/html/2407.14151v1 '''

    eval_freq = 10000 # Evaluate the model every 10,000 steps. This frequency allows you to monitor progress without interrupting training too often
    n_eval_episodes = 10  # Number of episodes for evaluation to obtain a reliable estimate of performance. https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html#how-to-evaluate-an-rl-algorithm

    ppo_eval_callback = EvalCallback(
        env,
        best_model_save_path=models_dirs["PPO"],
        log_path=logs_dirs["PPO"],
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )

    a2c_eval_callback = EvalCallback(
        env,
        best_model_save_path=models_dirs["A2C"],
        log_path=logs_dirs["A2C"],
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )

    dqn_eval_callback = EvalCallback(
        env,
        best_model_save_path=models_dirs["DQN"],
        log_path=logs_dirs["DQN"],
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )

    print("Training PPO model...")
    ppo_model.learn(total_timesteps=total_timesteps["PPO"], callback=[ProgressBarCallback(total_timesteps["PPO"]), ppo_eval_callback], log_interval=100)

    print("Training DQN model...")
    dqn_model.learn(total_timesteps=total_timesteps["DQN"], callback=[ProgressBarCallback(total_timesteps["DQN"]), dqn_eval_callback], log_interval=100)

    print("Training A2C model...")
    a2c_model.learn(total_timesteps=total_timesteps["A2C"], callback=[ProgressBarCallback(total_timesteps["A2C"]), a2c_eval_callback], log_interval=100)

    # Plot mean speeds
    if len(env.mean_speeds) > 0:
        plt.plot(env.mean_speeds)
        plt.xlabel("Step")
        plt.ylabel("Mean Speed")
        plt.title("Mean Speed over Training")
        plt.show()
    else:
        print("No mean speeds data to plot")

finally:
    env.close()