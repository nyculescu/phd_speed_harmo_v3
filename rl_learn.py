from traffic_environment.setup import create_sumocfg
from traffic_environment.flow_gen import *

import logging
import logging.handlers

from sb3_contrib import TRPO #, CrossQ, TQC
from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG #, DQN #, DroQ, DQN
from rl_models.custom_models.DDPG_PRDDPG import PRDDPG
from rl_models.custom_models.DoubleDQN import DoubleDQN as DQN
# from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback, BaseCallback, EveryNTimesteps
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from traffic_environment.rl_gym_environments import *
from traffic_environment.reward_functions import *
from traffic_environment.flow_gen import *
from gymnasium.wrappers import TimeLimit
import multiprocessing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import torch
# logging.info(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
from datetime import datetime 
from config import *
import time

# import platform
# logging.info(platform.architecture())

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
        if is_learning:
            env = Monitor(
                TrafficEnv(port=port, model=model, model_idx=model_idx, is_learning=is_learning, base_gen_car_distrib=["uniform", 2000])
                )
        else:
            env = Monitor(
                TimeLimit(
                    TrafficEnv(port=port, model=model, model_idx=model_idx, is_learning=is_learning, base_gen_car_distrib=["uniform", 3000]), 
                    max_episode_steps=episode_length * 60))
        return env
    return _init

def train_model(model_name, model, timesteps, callbacks):
    try:
        create_sumocfg(model_name, num_envs_per_model)

        try:
            logging.info(f"Starting model.learn() with timesteps={timesteps}")
            model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
        except Exception as e:
            logging.error(f"Error during training: {e.args}")
    except ValueError as e:
        logging.error(f"Error: {e.args}")

def train_dqn():
    timesteps = episode_length * num_of_hr_intervals * 60  # Total timesteps per phase
    model_name = 'DQN'
    log_dir = f"./logs/{model_name}/"
    model_dir = f"./rl_models/{model_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    create_sumocfg(model_name, num_envs_per_model)

    def train_env_constructor(port, is_learning):
        return get_traffic_env(port, model_name, model_idx=port - base_sumo_port, is_learning=is_learning)

    def eval_env_constructor(port, is_learning):
        return get_traffic_env(port, model_name, model_idx=0, is_learning=is_learning)

    train_env = SubprocVecEnv([
        train_env_constructor(base_sumo_port + idx, is_learning=True)
        for idx in range(num_envs_per_model)
    ])
    env_eval = SubprocVecEnv([
        eval_env_constructor(base_sumo_port + num_envs_per_model + 1, is_learning=False)
    ])

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,
        buffer_size=50000,
        batch_size=64,
        verbose=1,
        tensorboard_log=log_dir,
        device='cuda'
    )

    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    checkpoint_cb = CheckpointCallback(
        save_freq=timesteps,
        save_path=model_dir,
        name_prefix=f"rl_model_{model_name}",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1
    )

    no_improve_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=2,
        min_evals=2,
        verbose=1
    )

    eval_cb = EvalCallback(
        env_eval,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=timesteps,
        n_eval_episodes=2,
        deterministic=True,
        render=False,
        callback_after_eval=no_improve_cb,
        verbose=1
    )

    try:
        model.learn(total_timesteps=int(timesteps * num_of_iterations * num_envs_per_model), 
            callback=[checkpoint_cb, eval_cb], 
            progress_bar=True, 
            reset_num_timesteps=False
            )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        logging.error(f"Error during training: {e.args}")
    finally:
        # Clean up environments after training ends
        train_env.close()
        env_eval.close()

def train_ppo():
    model_name = 'PPO'
    log_dir = f"./logs/{model_name}/"

    model = PPO(
        "MlpPolicy",
        SubprocVecEnv([get_traffic_env(base_sumo_port + idx, model_name, idx, is_learning=True) 
                       for idx in range(num_envs_per_model)
                       ]),
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

    train_model(model_name, model)

def train_a2c():
    model_name = 'A2C'
    log_dir = f"./logs/{model_name}/"

    model = A2C("MlpPolicy", 
        SubprocVecEnv([get_traffic_env(base_sumo_port + idx, model_name, idx, is_learning=True) 
                       for idx in range(num_envs_per_model)
                       ]), 
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

    train_model(model_name, model)

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

    train_model(model_name, model)

def train_td3():
    model_name = 'TD3'
    log_dir = f"./logs/{model_name}/"

    model = TD3("MlpPolicy", 
        DummyVecEnv([get_traffic_env(base_sumo_port + cont_act_space_models.index(model_name), model_name, 0, is_learning = True)]),
        learning_rate=1e-3, 
        buffer_size=50000, 
        verbose=1, 
        tensorboard_log=log_dir,
        device='cuda', 
        batch_size=100, 
        policy_delay=2, 
        train_freq=(1, "episode"), 
        gradient_steps=-1)
        
    train_model(model_name, model)

def train_sac():
    model_name = 'SAC'
    log_dir = f"./logs/{model_name}/"

    model = SAC("MlpPolicy", 
        DummyVecEnv([get_traffic_env(base_sumo_port + cont_act_space_models.index(model_name), model_name, 0, is_learning = True)]),
        learning_rate=1e-3,  # Same as DQN
        buffer_size=50000,  # Same as DQN
        verbose=1,
        tensorboard_log=log_dir,
        device='cuda')

    train_model(model_name, model)

def train_ddpg():
    model_name = 'DDPG'
    log_dir = f"./logs/{model_name}/"

    # Create the model
    model = DDPG(
        "MlpPolicy",
        DummyVecEnv([get_traffic_env(base_sumo_port + cont_act_space_models.index(model_name), model_name, 0, is_learning = True)]),
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

    train_model(model_name, model)

def train_prddpg():
    model_name = 'PRDDPG'
    log_dir = f"./logs/{model_name}/"

    model = PRDDPG(
        "MlpPolicy",
        DummyVecEnv([get_traffic_env(base_sumo_port + cont_act_space_models.index(model_name), model_name, 0, is_learning = True)]),
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

    train_model(model_name, model)

# FIXME: This is used to log the results of the training process, but for now is a mock
def train_process_callback(result):
    logging.debug(f"Process finished with result: {result}")

def delayed_start(func, delay):
    time.sleep(delay)
    return func()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
    # Init
    multiprocessing.freeze_support()
    # set_full_day_car_generation_base_demand(250) # max no. of vehicles expected in any interval / hour
    # set_full_week_car_generation_rates(len(mock_daily_pattern(isFixed = True)) * len(day_of_the_week_factor))

    if not check_sumo_env():
        logging.info("SUMO environment is not set up correctly.") # FIXME: this is printed even if SUMO can run

    #######################################################################################
    # The training is split into 2 processes that shall run independently
    # # Training process 1
    # for m in discrete_act_space_models:
    #     for i in range(episodes):
    #         logging.info(f"Starting training for {m}, Episode {i+1}/{episodes}")
    #         # Train the model based on its name
    #         if m == 'TRPO':
    #             # train_trpo() # NOTE: trained
    #             pass
    #             logging.info(f"Skipping {m} training.")
    #         elif m == 'A2C':
    #             train_a2c()
    #             # logging.info(f"Skipping {m} training.")
    #         elif m == 'DQN':
    #             train_dqn()
    #             # logging.info(f"Skipping {m} training.")
    #         elif m == 'PPO':
    #             train_ppo()
    #             # logging.info(f"Skipping {m} training.")
    #         # sleep(2)
    #         gc.collect()
    
    #######################################################################################
    # # Training process 2 [Cover the constraint of AssertionError: You must use only one env when doing episodic training]
    # for i in range(episodes * num_envs_per_model):
    #     full_day_car_generation_base_demand_base -= 25
    #     set_full_day_car_generation_base_demand(full_day_car_generation_base_demand_base)

    #     # Create a pool of processes
    #     pool = multiprocessing.Pool(processes=3)

    #     # Collect async results
    #     async_results = [
    #         pool.apply_async(delayed_start, args=(train_td3, 0), callback=train_process_callback),
    #         pool.apply_async(delayed_start, args=(train_sac, 2), callback=train_process_callback),
    #         pool.apply_async(delayed_start, args=(train_ddpg, 4), callback=train_process_callback)
    #     ]

    #     # Close the pool and wait for all processes to finish
    #     logging.debug("Closing pool from RL_learn")
    #     pool.close()
    #     pool.join()

    #######################################################################################
    # # Training process 3
    train_dqn()
'''
Run from terminal (with .py310_tf_env activated): python -m memory_profiler rl_learn.py

Run rl_learn.py through tunnel:
/>  d:/phd_ws/speed_harmo/phd_speed_harmo_v3/.py310_tf_env/Scripts/python.exe d:/phd_ws/speed_harmo/phd_speed_harmo_v3/rl_learn.py
'''