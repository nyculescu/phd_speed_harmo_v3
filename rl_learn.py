from traffic_environment.setup import create_sumocfg
from traffic_environment.flow_gen import *

import logging
import logging.handlers

from stable_baselines import DQN
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback, BaseCallback
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from traffic_environment.rl_gym_environments import *
from traffic_environment.reward_functions import *
from traffic_environment.flow_gen import *
from gym.wrappers import TimeLimit
import multiprocessing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datetime import datetime 
from config import *

import tensorflow as tf

first_evaluation_length = int(episode_length * num_of_hr_intervals * 60 * 0.7)

class UpdateStartLaterCallback(BaseCallback):
    def __init__(self, eval_callback, verbose=0):
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.first_eval_done = False
        self.last_eval_done = False

    def _on_step(self):
        if self.last_eval_done == True:
            self.training_env.close()
            return False
        if not self.first_eval_done and self.eval_callback.n_calls >= first_evaluation_length:
            self.eval_callback.eval_freq = int(episode_length * num_of_hr_intervals * 60 * 0.1)
            self.first_eval_done = True
        if self.first_eval_done and self.eval_callback.n_calls >= (episode_length * num_of_hr_intervals * 60) :
            self.last_eval_done = True
        return True

def get_traffic_env(port, model, model_idx, is_learning):
    def _init():
        if is_learning:
            env = TrafficEnv(port=port, model=model, model_idx=model_idx, is_learning=is_learning, base_gen_car_distrib=["uniform", 300])
        else:
            env = TimeLimit(
                    TrafficEnv(port=port, model=model, model_idx=model_idx, is_learning=is_learning, base_gen_car_distrib=["uniform", 500]), 
                    max_episode_steps=episode_length * 60)
        return env
    return _init


def train_dqn():
    model_name = 'DQN'
    log_dir = f"./logs/{model_name}/"
    
    model = DQN("MlpPolicy", 
                SubprocVecEnv([get_traffic_env(base_sumo_port + idx, model_name, idx, is_learning=True) for idx in range(num_envs_per_model)]), 
                # learning_rate=1e-3, 
                buffer_size=50000, 
                verbose=1, 
                tensorboard_log=log_dir)
    
    log_dir = f"./logs/{model_name}/"
    model_dir = f"./rl_models/{model_name}/"
    timesteps = episode_length * num_of_hr_intervals * 60 # times aggregation time

    try:
        create_sumocfg(model_name, num_envs_per_model)
        if model_name in discrete_act_space_models:
            # The port is set as base + num_envs_per_model + 1 to avoid potential conflicts in case some SUMO processes are still running
            env_eval = SubprocVecEnv([get_traffic_env(base_sumo_port + num_envs_per_model + 1, model_name, 0, is_learning = False)])
        elif model_name in cont_act_space_models:
            env_eval = DummyVecEnv([get_traffic_env(base_sumo_port + len(cont_act_space_models) + 1, model_name, 0, is_learning = False)])
        else:
            raise ValueError(f"Model '{model_name}' not found in the models list.")
        
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # no_improve_cb = StopTrainingOnNoModelImprovement(
        #     max_no_improvement_evals=2,
        #     min_evals=2,
        #     verbose=1
        # )

        checkpoint_cb = CheckpointCallback(
            save_freq=episode_length * 60,
            save_path=model_dir,
            name_prefix=f"rl_model_{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            # save_replay_buffer=True,
            # save_vecnormalize=True,
            verbose=1
        )
        
        eval_cb = EvalCallback(
            env_eval,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=first_evaluation_length, # num_envs_per_model * (eval_freq * 7) = num_timesteps: a day could be enough for testing
            n_eval_episodes=2,
            deterministic=True,
            render=False,
            # callback_after_eval = no_improve_cb,
            verbose=1
        )
        
        update_start_later_cb = UpdateStartLaterCallback(eval_cb)
        
        try:
            logging.info(f"Starting model.learn() with timesteps={timesteps}")
            model.learn(total_timesteps=timesteps, callback=[eval_cb, checkpoint_cb, update_start_later_cb], tb_log_name=model_name)
        except Exception as e:
            logging.error(f"Error during training: {e.args}")
    except ValueError as e:
        logging.error(f"Error: {e.args}")

if __name__ == '__main__':
    set_full_day_car_generation_base_demand(250) # max no. of vehicles expected in any interval / hour
    set_full_week_car_generation_rates(len(mock_daily_pattern(isFixed = True)) * len(day_of_the_week_factor))

    if not check_sumo_env():
        logging.info("SUMO environment is not set up correctly.") # FIXME: this is printed even if SUMO can run

    print(tf.test.is_gpu_available())

    train_dqn()