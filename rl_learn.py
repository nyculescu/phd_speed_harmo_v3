from matplotlib import pyplot as plt
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import logging
import logging.handlers
from rl_gym_environments import SUMOEnv

from stable_baselines3.common.callbacks import BaseCallback
import multiprocessing
from time import sleep
from simulation_utilities.setup import create_sumocfg

from simulation_utilities.flow_gen import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
logging.info(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

'''
Factors to Consider to determine the appropriate number of total timesteps
1. Environment Complexity: SUMOEnv simulates traffic scenarios with a discrete action space of 11 actions, representing speed limits. The observation space is a simple 1-dimensional Box, indicating a relatively straightforward state representation. This simplicity suggests that fewer timesteps might be needed compared to more complex environments.
2. Simulation Length: Each episode in the environment runs for 120 steps (sim_length is set to 3600 divided by aggregation_time of 30). This means that the agent has 120 opportunities to learn from each episode.
3. Desired Performance: If the goal is to achieve a basic level of performance or to test feasibility, starting with fewer timesteps is reasonable. High performance -> more timesteps
4. Computational Resources: The availability of computational power can limit or extend the feasible number of timesteps. Training on GPUs can significantly speed up the process.
'''
''' Initialize models with recommended hyperparameters 
Note about policies:
    - Use "MlpPolicy" for environments with continuous state spaces, as it provides a robust architecture for handling complex observations
    - Use "MlpPolicy" for vector inputs or "CnnPolicy" if dealing with image-based inputs
    Ref.: De La Fuente, N., & Guerra, D. A. V. (2024). A Comparative Study of Deep Reinforcement Learning Models: DQN vs PPO vs A2C. arXiv preprint arXiv:2407.14151. -> link: https://arxiv.org/html/2407.14151v1
'''
  
''' Learning rate values: https://arxiv.org/html/2407.14151v1 '''

def make_env(port, model, model_idx):
    def _init():
        return SUMOEnv(port=port, model=model, model_idx=model_idx)
    return _init

def plot_mean_speeds(mean_speeds, agent_name):
    if len(mean_speeds) > 0:
        plt.figure()
        plt.plot(mean_speeds)
        plt.xlabel("Step")
        plt.ylabel("Mean Speed")
        plt.title(f"Mean Speed over Training for {agent_name}")
        plt.show()
    else:
        logging.debug(f"No mean speeds data to plot for {agent_name}")

class LoggingCallback(BaseCallback):
    def __init__(self, custom_logger, total_timesteps, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.custom_logger = custom_logger
        self.total_timesteps = total_timesteps
        self.one_percent_steps = total_timesteps // 100  # Calculate steps for 1%
        self.last_logged_step = 0

    def _on_step(self) -> bool:
        # Check if we've reached the next 1% step
        if (self.num_timesteps - self.last_logged_step) >= self.one_percent_steps:
            self.custom_logger.info(f"Progress: {self.num_timesteps / self.total_timesteps:.2%} completed.")
            self.last_logged_step = self.num_timesteps
        return True

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)        
    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

eval_freq = 10000 # Evaluate the model every 10,000 steps. This frequency allows you to monitor progress without interrupting training too often
n_eval_episodes = 10  # Number of episodes for evaluation to obtain a reliable estimate of performance. https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html#how-to-evaluate-an-rl-algorithm

# Define the number of environments per model
num_envs_per_model = 5
base_port = 8810
models = ["DQN", "A2C", "PPO"]
total_timesteps = 500000

def train_ppo():
    model_name = 'PPO'
    ports = [base_port + i for i in range(num_envs_per_model)]
    env = SubprocVecEnv([make_env(port, model_name, idx) for idx, port in enumerate(ports)])

    log_dir = f"./logs/{model_name}/"
    model_dir = f"./rl_models/{model_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Initialize PPO model with vectorized environments
    ppo_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=log_dir,
        device='cuda'
    )

    ppo_eval_callback = EvalCallback(
        env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # Set up custom logger
    ppo_logger = setup_logger(f'{model_name}', f'{log_dir}/{model_name}_training.log')

    # Initialize LoggingCallback
    logging_callback = LoggingCallback(ppo_logger, total_timesteps)

    logging.info(f"Training {model_name} model...")
    ppo_model.learn(total_timesteps=total_timesteps, callback=[ppo_eval_callback, logging_callback])

def train_dqn():
    model_name = 'DQN'
    ports = [(base_port + num_envs_per_model) + i for i in range(num_envs_per_model)]
    env = SubprocVecEnv([make_env(port, model_name, idx) for idx, port in enumerate(ports)])

    log_dir = f"./logs/{model_name}/"
    model_dir = f"./rl_models/{model_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    dqn_model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
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
        tensorboard_log=log_dir,
        device='cuda'
    )

    dqn_eval_callback = EvalCallback(
        env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # Set up custom logger
    ppo_logger = setup_logger(f'{model_name}', f'{log_dir}/{model_name}_training.log')

    # Initialize LoggingCallback
    logging_callback = LoggingCallback(ppo_logger, total_timesteps)

    logging.info(f"Training {model_name} model...")
    dqn_model.learn(total_timesteps=total_timesteps, callback=[dqn_eval_callback, logging_callback])

def train_a2c():
    model_name = 'A2C'
    ports = [(base_port + num_envs_per_model * 2) + i for i in range(num_envs_per_model)]
    env = SubprocVecEnv([make_env(port, model_name, idx) for idx, port in enumerate(ports)])

    log_dir = f"./logs/{model_name}/"
    model_dir = f"./rl_models/{model_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    a2c_model = A2C(
        "MlpPolicy",
        env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        device='cuda'
    )

    a2c_eval_callback = EvalCallback(
        env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # Set up custom logger
    ppo_logger = setup_logger(f'{model_name}', f'{log_dir}/{model_name}_training.log')

    # Initialize LoggingCallback
    logging_callback = LoggingCallback(ppo_logger, total_timesteps)

    logging.info(f"Training {model_name} model...")
    a2c_model.learn(total_timesteps=total_timesteps, callback=[a2c_eval_callback, logging_callback])

if __name__ == '__main__':
    create_sumocfg(models, num_envs_per_model)

    # Ensure freeze_support() is called if necessary (typically for Windows)
    multiprocessing.freeze_support()

    # Create a process for each training function
    ppo_process = multiprocessing.Process(target=train_ppo)
    dqn_process = multiprocessing.Process(target=train_dqn)
    a2c_process = multiprocessing.Process(target=train_a2c)
    
    # Start the processes
    ppo_process.start()
    dqn_process.start()
    a2c_process.start()
    
    # Join the processes to ensure they complete before exiting
    ppo_process.join()
    dqn_process.join()
    a2c_process.join()

    # train_ppo() # FIXME: this one is called only for debugging purposes

'''
Run rl_learn.py through tunnel:
/>  d:/phd_ws/speed_harmo/phd_speed_harmo_v3/.py310_tf_env/Scripts/python.exe d:/phd_ws/speed_harmo/phd_speed_harmo_v3/rl_learn.py
'''