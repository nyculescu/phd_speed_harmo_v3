from simulation_utilities.setup import create_sumocfg
from simulation_utilities.flow_gen import *

from matplotlib import pyplot as plt
import multiprocessing
import logging
import logging.handlers

from sb3_contrib import TRPO #, CrossQ, TQC
from stable_baselines3 import PPO, DQN, A2C, SAC, TD3 #, DroQ
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from rl_gym_environments import *
from rl_utilities.reward_functions import *
from simulation_utilities.flow_gen import *
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
logging.info(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
from rl_zoo3.train import train

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

global_day_index = 0

def make_env_with_monitor(port, model, model_idx):
    def _init():
        env = SUMOEnv(port=port, model=model, model_idx=model_idx)
        if model_idx == 0:
            check_env(env) # useful when there are multiple reward functions
        env.day_index = (global_day_index + model_idx) % 7
        return Monitor(env)
    return _init

def make_env(port, model, model_idx):
    def _init():
        env = SUMOEnv(port=port, model=model, model_idx=model_idx)
        env.day_index = (global_day_index + model_idx) % 7
        return env  # No Monitor
    return _init

def start_process_with_delay(processes, delay_seconds):
    last_start_time = time.time()

    for process in processes:
        # Wait until the required delay has passed
        while time.time() - last_start_time < delay_seconds:
            time.sleep(0.1)  # Sleep briefly to prevent busy-waiting

        # Start the process
        process.start()
        print(f"Started {process.name} at {time.time()}")

        # Update the last start time
        last_start_time = time.time()

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

class ShowProgressCallback(BaseCallback):
    def __init__(self, custom_logger, total_timesteps, verbose=0):
        super(ShowProgressCallback, self).__init__(verbose)
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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Configuration """

# eval_freq = 10000 # Evaluate the model every 10,000 steps. This frequency allows you to monitor progress without interrupting training too often
# n_eval_episodes = 10  # Number of episodes for evaluation to obtain a reliable estimate of performance. https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html#how-to-evaluate-an-rl-algorithm

base_port = 8800
# total_timesteps = 50000

def train_ppo():
    model_name = 'PPO'
    log_dir = f"./logs/{model_name}/"
    model_dir = f"./rl_models/{model_name}/"
    timesteps = (len(car_generation_rates_per_lane) / 2) * 60

    # Get the index of the model in the list
    try:
        create_sumocfg(model_name, num_envs_per_model)
        ports = [(base_port + num_envs_per_model) + i for i in range(num_envs_per_model)]
        env = SubprocVecEnv([make_env_with_monitor(port, model_name, idx) for idx, port in enumerate(ports)])
        env_eval = SubprocVecEnv([make_env(port, model_name, idx) for idx, port in enumerate(ports[:7])])

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # Save a checkpoint every 1000 steps
        checkpoint_cb = CheckpointCallback(
            save_freq=1000,
            save_path=model_dir,
            name_prefix=f"rl_model_{model_name}",
            save_replay_buffer=True,
            save_vecnormalize=True,
            )

        # Initialize PPO model with vectorized environments
        model = PPO(
            "MlpPolicy",
            env_eval,
            learning_rate=2e-4, # old: 3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=15,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2, # reduced from 0.2 to prevent large policy updates
            ent_coef=0.02, # start with 0.01 to encourage more exploration -> help prevent the model from converging prematurely to suboptimal policies
            verbose=1,
            tensorboard_log=log_dir,
            device='cuda'
        )
        
        # Stop training if there is no improvement after more than 3 evaluations
        stop_train_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
        eval_cb = EvalCallback(
            env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=max(10000 // num_envs_per_model, 1),
            callback_after_eval=stop_train_cb,
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
            render=False
        )

        model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

        logging.info(f"Training {model_name} model...")
        model.learn(total_timesteps=timesteps, callback=[checkpoint_cb, eval_cb], progress_bar=True)
    except ValueError:
        print(f"Warning: Model '{model_name}' not found in the models list. Skipping...")

def train_dqn():
    model_name = 'DQN'
    log_dir = f"./logs/{model_name}/"
    model_dir = f"./rl_models/{model_name}/"
    compl_logger = setup_logger(f'{model_name}', f'{log_dir}/{model_name}_training.log')
    timesteps = (len(car_generation_rates_per_lane) / 2) * 60
    try:
        create_sumocfg(model_name, num_envs_per_model)
        ports = [(base_port + num_envs_per_model) + i for i in range(num_envs_per_model)]
        env_mon = SubprocVecEnv([make_env_with_monitor(port, model_name, idx) for idx, port in enumerate(ports)])
        env_eval = SubprocVecEnv([make_env(port, model_name, idx) for idx, port in enumerate(ports[:7])])

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        model = DQN(
            "MlpPolicy",
            env_mon,
            learning_rate=2e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=128, # Increase batch size to stabilize training updates
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            target_update_interval=500, # Old: reduced from 1000 for more frequent updates
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log=log_dir,
            device='cuda'
        )
        """
        Notes:
        1. use Prioritized Replay Buffer (PER) to focus on more informative experiences.
        """

        eval_callback = EvalCallback(
            env_eval,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=max(10000 // num_envs_per_model, 1),
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )

        model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

        progress_callback = ShowProgressCallback(compl_logger, timesteps)

        logging.info(f"Training {model_name} model...")
        model.learn(total_timesteps=timesteps, callback=[eval_callback, progress_callback])
    except ValueError:
        print(f"Warning: Model '{model_name}' not found in the models list. Skipping...")

def train_a2c():
    model_name = 'A2C'
    log_dir = f"./logs/{model_name}/"
    model_dir = f"./rl_models/{model_name}/"
    compl_logger = setup_logger(f'{model_name}', f'{log_dir}/{model_name}_training.log')
    timesteps = (len(car_generation_rates_per_lane) / 2) * 60
    create_sumocfg(model_name, num_envs_per_model)
    ports = [(base_port + num_envs_per_model * 2) + i for i in range(num_envs_per_model)]
    env = SubprocVecEnv([make_env_with_monitor(port, model_name, idx) for idx, port in enumerate(ports)])

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=5e-4, # trained before with 7e-4
        n_steps=10, # increased from 5 to capture more information per update
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        device='cuda'
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    progress_callback = ShowProgressCallback(compl_logger, timesteps)

    logging.info(f"Training {model_name} model...")
    model.learn(total_timesteps=timesteps, callback=[eval_callback, progress_callback])

def train_trpo():
    model_name = 'TRPO'
    log_dir = f"./logs/{model_name}/"
    model_dir = f"./rl_models/{model_name}/"
    compl_logger = setup_logger(f'{model_name}', f'{log_dir}/{model_name}_training.log')
    timesteps = (len(car_generation_rates_per_lane) / 2) * 60
    try:
        create_sumocfg(model_name, num_envs_per_model)
        ports = [(base_port + num_envs_per_model) + i for i in range(num_envs_per_model)]
        env = SubprocVecEnv([make_env_with_monitor(port, model_name, idx) for idx, port in enumerate(ports)])
    
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize TRPO model with vectorized environments
        model = TRPO(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            n_steps=2048,
            gamma=0.99,
            verbose=1,
            tensorboard_log=log_dir,
            device='cuda'
        )
        
        # Set up evaluation callback
        eval_callback = EvalCallback(
            env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=10000,
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )
        
        model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))
        progress_callback = ShowProgressCallback(compl_logger, timesteps)

        logging.info(f"Training {model_name} model...")
        model.learn(total_timesteps=timesteps, callback=[eval_callback, progress_callback])
    except ValueError:
        print(f"Warning: Model '{model_name}' not found in the models list. Skipping...")

def train_td3():
    model_name = 'TD3'
    log_dir = f"./logs/{model_name}/"
    model_dir = f"./rl_models/{model_name}/"
    compl_logger = setup_logger(f'{model_name}', f'{log_dir}/{model_name}_training.log')
    timesteps = (len(car_generation_rates_per_lane) / 2) * 60
    try:
        create_sumocfg(model_name, num_envs_per_model)
        ports = [(base_port + num_envs_per_model) + i for i in range(num_envs_per_model)]
        env = SubprocVecEnv([make_env_with_monitor(port, model_name, idx) for idx, port in enumerate(ports)])

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize TD3 model with the single environment
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            buffer_size=1000000,
            learning_starts=10000,
            batch_size=100,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),  # Use "step" to ensure correct training frequency
            gradient_steps=-1,
            verbose=1,
            tensorboard_log=log_dir,
            device='cuda'
        )
        
        # Set up evaluation callback
        eval_callback = EvalCallback(
            env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=10000,
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )

        model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))
        progress_callback = ShowProgressCallback(compl_logger, timesteps)

        logging.info(f"Training {model_name} model...")
        model.learn(total_timesteps=timesteps, callback=[eval_callback, progress_callback])
    except ValueError:
        print(f"Warning: Model '{model_name}' not found in the models list. Skipping...")

def train_sac():
    model_name = 'SAC'
    log_dir = f"./logs/{model_name}/"
    model_dir = f"./rl_models/{model_name}/"
    compl_logger = setup_logger(f'{model_name}', f'{log_dir}/{model_name}_training.log')
    timesteps = (len(car_generation_rates_per_lane) / 2) * 60
    try:
        create_sumocfg(model_name, num_envs_per_model)
        """ A singe instance is allowed for SAC because of this error: 
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."""
        port = base_port + num_envs_per_model
        env = DummyVecEnv([make_env_with_monitor(port, model_name, 0)])  # Use DummyVecEnv for single env

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=10000,
            batch_size=256, # Increase for more stable updates
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "episode"),
            gradient_steps=10, # increase for more thorough learning per update
            ent_coef='auto', # automatic entropy tuning
            verbose=1,
            tensorboard_log=log_dir,
            device='cuda'
        )

        eval_callback = EvalCallback(
            env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=10000,
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )

        model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))
        progress_callback = ShowProgressCallback(compl_logger, timesteps)

        logging.info(f"Training {model_name} model...")
        model.learn(total_timesteps=timesteps, callback=[eval_callback, progress_callback])
    except ValueError:
        print(f"Warning: Model '{model_name}' not found in the models list. Skipping...")

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
    episodes = 1
    
    for i in range(episodes):
        train_ppo()
        global_day_index = (global_day_index + num_envs_per_model) % 7
        
    # train_a2c()
    # train_dqn()
    # train_sac()
    # train_td3()
    # train_trpo()

'''
Run rl_learn.py through tunnel:
/>  d:/phd_ws/speed_harmo/phd_speed_harmo_v3/.py310_tf_env/Scripts/python.exe d:/phd_ws/speed_harmo/phd_speed_harmo_v3/rl_learn.py
'''