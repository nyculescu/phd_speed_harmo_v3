from traffic_environment.setup import create_sumocfg
from traffic_environment.flow_gen import *

import logging
import logging.handlers

from sb3_contrib import TRPO #, CrossQ, TQC
from stable_baselines3 import PPO, DQN, A2C, SAC, TD3, DDPG #, DroQ
# from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from traffic_environment.rl_gym_environments import *
from traffic_environment.reward_functions import *
from traffic_environment.flow_gen import *
# import gymnasium as gym
import multiprocessing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
logging.info(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
from datetime import datetime 
from config import *

# rl_zoo3 training script to train an agent /> python -m rl_zoo3.train --algo ppo --env TrafficEnv-v0 --eval-freq 10000 --save-freq 50000 --n-timesteps 1000000
# Hyperparameter Optimization (Optuna with rl_zoo3) /> python -m rl_zoo3.train --algo dqn --env TrafficEnv-v0 -n 50000 -optimize --n-trials 1000 --n-jobs 2 --sampler random --pruner median

'''
Factors to Consider to determine the appropriate number of total timesteps
1. Environment Complexity: TrafficEnv simulates traffic scenarios with a discrete action space of 11 actions, representing speed limits. The observation space is a simple 1-dimensional Box, indicating a relatively straightforward state representation. This simplicity suggests that fewer timesteps might be needed compared to more complex environments.
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

def get_traffic_env(port, model, model_idx, is_learning):
    def _init():
        env = Monitor(TrafficEnv(port=port, model=model, model_idx=model_idx, is_learning=is_learning))
        # check_env(env, warn=True)

        # env.update_environment()  # Simulate a change in road conditions
        
        return env
    return _init

# n_eval_episodes = 10  # Number of episodes for evaluation to obtain a reliable estimate of performance. https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html#how-to-evaluate-an-rl-algorithm

def train_model(model_name, model, ports):
    log_dir = f"./logs/{model_name}/"
    model_dir = f"./rl_models/{model_name}/"
    episode_length = car_generation_rates * 60
    timesteps = episode_length * num_envs_per_model

    try:
        create_sumocfg(model_name, num_envs_per_model)
        if model_name in discrete_act_space_models:
            env_eval = SubprocVecEnv([get_traffic_env(base_sumo_port + len(ports) + i, model_name, i, is_learning = False) for i in len(ports)])
        elif model_name in cont_act_space_models:
            env_eval = DummyVecEnv([get_traffic_env(ports[0] + len(cont_act_space_models), model_name, 0, is_learning = False)])
        
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        no_improve_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=2,
            min_evals=2,
            verbose=1
        )

        checkpoint_cb = CheckpointCallback(
            save_freq=episode_length,
            save_path=model_dir,
            name_prefix=f"rl_model_{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            save_replay_buffer=True,
            save_vecnormalize=True,
            verbose=1
        )
        
        eval_cb = EvalCallback(
            env_eval,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=episode_length, # num_envs_per_model * (eval_freq * 7) = num_timesteps: a day could be enough for testing
            n_eval_episodes=2,
            deterministic=True,
            render=False,
            callback_after_eval = no_improve_cb,
            verbose=1
        )
        
        model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

        logging.info(f"Training {model_name} model...")
        model.learn(total_timesteps=timesteps, callback=[eval_cb, checkpoint_cb], progress_bar=True)
    except ValueError as e:
        print(f"Warning: Model '{model_name}' not found in the models list. Skipping...")
        print(f"Error: {e}")

def train_ppo():
    model_name = 'PPO'
    log_dir = f"./logs/{model_name}/"
    ports = [(base_sumo_port + i) for i in range(num_envs_per_model)]

    model = PPO(
        "MlpPolicy",
        SubprocVecEnv([get_traffic_env(port, model_name, idx, is_learning = True) for idx, port in enumerate(ports)]),
        learning_rate=1e-3,  # Similar to DQN for comparative analysis
        n_steps=2048,        # Number of steps to run for each environment per update
        batch_size=64,       # Batch size for each update
        n_epochs=10,         # Number of epochs when optimizing the surrogate loss
        gamma=0.99,          # Discount factor
        gae_lambda=0.95,     # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        clip_range=0.2,      # Clipping parameter for PPO
        ent_coef=0.01,       # Entropy coefficient for exploration
        verbose=1,
        tensorboard_log=log_dir,
        device='cuda'
    )

    train_model(model_name, model, ports)

def train_dqn():
    model_name = 'DQN'
    ports = [(base_sumo_port + i) for i in range(num_envs_per_model)]
    log_dir = f"./logs/{model_name}/"
    # model = DQN(
    #     "MlpPolicy",
    #     env_mon,
    #     learning_rate=2e-4,
    #     buffer_size=100000,
    #     learning_starts=1000,
    #     batch_size=128, # Increase batch size to stabilize training updates
    #     tau=1.0,
    #     gamma=0.99,
    #     train_freq=4,
    #     target_update_interval=500, # Old: reduced from 1000 for more frequent updates
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.05,
    #     verbose=1,
    #     tensorboard_log=log_dir,
    #     device='cuda'
    # )
    model = DQN("MlpPolicy", SubprocVecEnv([get_traffic_env(port, model_name, idx, is_learning = True) for idx, port in enumerate(ports)]), 
                learning_rate=1e-3, buffer_size=50000, verbose=1, tensorboard_log=log_dir,
                device='cuda')
    """
    Notes:
    1. use Prioritized Replay Buffer (PER) to focus on more informative experiences.
    """
    train_model(model_name, model, ports)

def train_a2c():
    model_name = 'A2C'
    log_dir = f"./logs/{model_name}/"
    ports = [(base_sumo_port + i) for i in range(num_envs_per_model)]

    model = A2C("MlpPolicy", 
        SubprocVecEnv([get_traffic_env(port, model_name, idx, is_learning = True) for idx, port in enumerate(ports)]), 
        learning_rate=1e-3, 
        n_steps=5,  # Adjust based on your environment's needs
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE parameter
        ent_coef=0.01,  # Entropy coefficient
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        device='cuda')

    train_model(model_name, model, ports)

def train_trpo():
    model_name = 'TRPO'
    log_dir = f"./logs/{model_name}/"
    ports = [(base_sumo_port + i) for i in range(num_envs_per_model)]

    model = TRPO("MlpPolicy", 
        SubprocVecEnv([get_traffic_env(port, model_name, idx, is_learning = True) for idx, port in enumerate(ports)]), 
        learning_rate=1e-3, 
        verbose=1, 
        tensorboard_log=log_dir, 
        device='cuda')

    train_model(model_name, model, ports)

def train_td3(): # NOTE: is has progress bar
    model_name = 'TD3'
    log_dir = f"./logs/{model_name}/"
    sumo_port = base_sumo_port + cont_act_space_models.index(model_name)
    model = TD3("MlpPolicy", 
        DummyVecEnv([get_traffic_env(sumo_port, model_name, 0, is_learning = True)]),
        learning_rate=1e-3, 
        buffer_size=50000, 
        verbose=1, 
        tensorboard_log=log_dir,
        device='cuda', 
        batch_size=100, 
        policy_delay=2, 
        train_freq=(1, "episode"), 
        gradient_steps=-1)
        
    train_model(model_name, model, [sumo_port])

def train_sac():
    model_name = 'SAC'
    log_dir = f"./logs/{model_name}/"
    sumo_port = base_sumo_port + cont_act_space_models.index(model_name)

    model = SAC("MlpPolicy", 
        DummyVecEnv([get_traffic_env(sumo_port, model_name, 0, is_learning = True)]), 
        learning_rate=1e-3,  # Same as DQN
        buffer_size=50000,  # Same as DQN
        verbose=1,
        tensorboard_log=log_dir,
        device='cuda')

    train_model(model_name, model, [sumo_port])

def train_ddpg():
    model_name = 'DDPG'
    log_dir = f"./logs/{model_name}/"
    sumo_port = base_sumo_port + cont_act_space_models.index(model_name)

    # Create the model
    model = DDPG(
        "MlpPolicy",
        DummyVecEnv([get_traffic_env(sumo_port, model_name, 0, is_learning = True)]),
        learning_rate=1e-3,
        buffer_size=50000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        train_freq=(1, 'episode'),
        gradient_steps=-1,
        verbose=1,
        tensorboard_log=log_dir,
        device='cuda'
    )

    train_model(model_name, model, [sumo_port])

# FIXME: This is used to log the results of the training process, but for now is a mock
def train_process_callback(result):
    logging.debug(f"Process finished with result: {result}")

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
    multiprocessing.freeze_support()
    episodes = 5

    # The training is splin into 2 processes that shall run independently
    # Training process 1
    # for i in range(episodes):
        # train_ppo()
        # train_a2c()
        # train_dqn()
        # train_trpo()
    
    # Training process 2
    # Cover the constraint of AssertionError: You must use only one env when doing episodic training
    for i in range(episodes * num_envs_per_model):
        # Create a pool of processes
        pool = multiprocessing.Pool(processes=len(cont_act_space_models))

        # Collect async results
        async_results = [
            pool.apply_async(train_td3, callback=train_process_callback),
            pool.apply_async(train_sac, callback=train_process_callback),
            pool.apply_async(train_ddpg, callback=train_process_callback)
        ]

        # Close the pool and wait for all processes to finish
        logging.debug("Closing pool")
        pool.close()
        pool.join()
        

'''
Run rl_learn.py through tunnel:
/>  d:/phd_ws/speed_harmo/phd_speed_harmo_v3/.py310_tf_env/Scripts/python.exe d:/phd_ws/speed_harmo/phd_speed_harmo_v3/rl_learn.py
'''