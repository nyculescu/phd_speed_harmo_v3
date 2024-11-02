import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sb3_contrib import TRPO
from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
# from stable_baselines3.common.env_checker import check_env
from scipy import stats
import logging
import os
import multiprocessing
from time import sleep
from traffic_environment.flow_gen import flow_generation_wrapper
from traci import close as traci_close
from traci.exceptions import FatalTraCIError, TraCIException
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

from traffic_environment.rl_gym_environments import TrafficEnv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the log level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[  # Handlers determine where logs are sent
        logging.StreamHandler()  # Output logs to stderr (default)
    ]
)

test_without_electric = False
test_without_disobedient = False

# Define a custom TensorBoard callback that accepts an environment and model
class TensorboardCallback(BaseCallback):
    def __init__(self, env, model, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.env = env  # Store the environment
        self.model = model  # Store the model

    def _on_step(self) -> bool:
        # Access metrics from the environment
        reward = self.locals['rewards']  # Access rewards from the environment
        emissions = self.env.emissions_over_time[-1]  # Log latest emissions data
        mean_speed = self.env.mean_speed_over_time[-1]  # Log latest mean speed data
        flow = self.env.flows[-1]  # Log latest flow data
        
        # Record these values in TensorBoard using SB3's built-in logger
        self.logger.record('test/reward', reward)
        self.logger.record('test/emissions', emissions)
        self.logger.record('test/mean_speed', mean_speed)
        self.logger.record('test/flow', flow)
        
        return True  # Continue running the environment

def reward_filter_flat_lines(rewards, threshold=1e-2, flat_line_threshold=10):
    # Compute differences between consecutive rewards
    diffs = np.abs(np.diff(rewards))
    
    # Identify where differences are below the threshold
    near_zero_diffs = diffs < threshold
    
    # Create a mask to keep only relevant points
    mask = np.ones(len(rewards), dtype=bool)
    
    # Loop through and find "flat lines" (consecutive near-zero differences)
    count_flat = 0
    for i in range(1, len(near_zero_diffs)):
        if near_zero_diffs[i-1]:
            count_flat += 1
        else:
            count_flat = 0
        
        # If we have a flat line longer than the threshold, start filtering out points
        if count_flat >= flat_line_threshold:
            mask[i+1] = False
    
    # Apply the mask to filter out the irrelevant points
    filtered_rewards = np.array(rewards)[mask]
    filtered_indices = np.arange(len(rewards))[mask]
    
    return filtered_indices, filtered_rewards

# Suppress matplotlib debug output
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

model_paths = {
    "PPO": "rl_models/PPO/best_model",
    "A2C": "rl_models/A2C/best_model",
    "DQN": "rl_models/DQN/best_model",
    "TRPO": "rl_models/TRPO/best_model",
    "TD3": "rl_models/TD3/best_model",
    "SAC": "rl_models/SAC/best_model"
}

# Define colors for each agent
colors = {
    'PPO': 'grey',
    'A2C': 'violet',
    'DQN': 'turquoise',
    'TRPO': 'lightgreen',
    'TD3': 'khaki',
    'SAC': 'chocolate'
}
ports = {'PPO': 8810, 'A2C': 8811, 'DQN': 8812, 'TRPO': 8813, 'TD3': 8814, 'SAC': 8815}

results = {}

metrics_to_plot = ['rewards'
                #    , 'obs'
                   , 'emissions'
                   , 'mean speed'
                   , 'flow'
                   ]

""" Metric Storage and Analysis """
def save_metrics(metrics, agent_name):
    # Open the file using a context manager
    with open(f'./logs/{agent_name}_metrics.csv', 'w+') as metrics_file:
        # Iterate over each key and write its corresponding values
        for key in metrics_to_plot:
            # Convert list to comma-separated string and write to file
            metrics_file.write(','.join(map(str, metrics[key])) + '\n')
    
    # pd.DataFrame(metrics['cvs_seg_time']).to_csv(f'./metrics/{agent_name}.csv', index=False, header=False)


def test_ppo():
    model_name = "PPO"
    model = PPO.load(model_paths[model_name])
    # test_model(model, model_name)
    logging.debug(f"Starting {model_name} test")
    env = TrafficEnv(port=ports[model_name], model=model_name, model_idx=0)
    env.is_learning = False
    env.test_without_electric = test_without_electric
    env.test_without_disobedient = test_without_disobedient
    # check_env(env)
    
    # Set up TensorBoard logger
    tf_logger = configure(f"./tensorboard_logs/{model_name}_test", ["tensorboard"])
    model.set_logger(tf_logger)

    obs, _ = env.reset()
    done = False

    rewards = []
    # Initialize custom callback for logging with the environment passed in
    tensorboard_callback = TensorboardCallback(env, model)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        
        # Log reward and other metrics to TensorBoard using the callback
        tensorboard_callback.locals = {'rewards': reward}
        tensorboard_callback._on_step()  # Manually call _on_step() to log metrics
        
        rewards.append(reward)
    
    tf_logger.dump(step=0)  # Ensure logs are written

    metrics = [rewards, env.emissions_over_time, env.mean_speed_over_time, env.flows]
    result = {metric: value for metric, value in zip(metrics_to_plot, metrics)}

    return (model_name, result)

def test_a2c():
    model_name = "A2C"
    model = A2C.load(model_paths[model_name])
    # test_model(model, model_name)
    logging.debug(f"Starting {model_name} test")
    env = TrafficEnv(port=ports[model_name], model=model_name, model_idx=0)
    env.is_learning = False
    env.test_without_electric = test_without_electric
    env.test_without_disobedient = test_without_disobedient
    # check_env(env)
    
    # Set up TensorBoard logger
    tf_logger = configure(f"./tensorboard_logs/{model_name}_test", ["tensorboard"])
    model.set_logger(tf_logger)

    obs, _ = env.reset()
    done = False

    rewards = []
    # Initialize custom callback for logging with the environment passed in
    tensorboard_callback = TensorboardCallback(env, model)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        
        # Log reward and other metrics to TensorBoard using the callback
        tensorboard_callback.locals = {'rewards': reward}
        tensorboard_callback._on_step()  # Manually call _on_step() to log metrics
        
        rewards.append(reward)
    
    tf_logger.dump(step=0)  # Ensure logs are written

    metrics = [rewards, env.emissions_over_time, env.mean_speed_over_time, env.flows]
    result = {metric: value for metric, value in zip(metrics_to_plot, metrics)}

    return (model_name, result)

def test_dqn():
    model_name = "DQN"
    model = DQN.load(model_paths[model_name])
    # test_model(model, model_name)
    logging.debug(f"Starting {model_name} test")
    env = TrafficEnv(port=ports[model_name], model=model_name, model_idx=0)
    env.is_learning = False
    env.test_without_electric = test_without_electric
    env.test_without_disobedient = test_without_disobedient
    # check_env(env)
    
    # Set up TensorBoard logger
    tf_logger = configure(f"./tensorboard_logs/{model_name}_test", ["tensorboard"])
    model.set_logger(tf_logger)

    obs, _ = env.reset()
    done = False

    rewards = []
    # Initialize custom callback for logging with the environment passed in
    tensorboard_callback = TensorboardCallback(env, model)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        
        # Log reward and other metrics to TensorBoard using the callback
        tensorboard_callback.locals = {'rewards': reward}
        tensorboard_callback._on_step()  # Manually call _on_step() to log metrics
        
        rewards.append(reward)
    
    tf_logger.dump(step=0)  # Ensure logs are written

    metrics = [rewards, env.emissions_over_time, env.mean_speed_over_time, env.flows]
    result = {metric: value for metric, value in zip(metrics_to_plot, metrics)}

    return (model_name, result)

def test_td3():
    model_name = "TD3"
    model = TD3.load(model_paths[model_name])
    # test_model(model, model_name)
    logging.debug(f"Starting {model_name} test")
    env = TrafficEnv(port=ports[model_name], model=model_name, model_idx=0)
    env.is_learning = False
    env.test_without_electric = test_without_electric
    env.test_without_disobedient = test_without_disobedient
    # check_env(env)
    
    # Set up TensorBoard logger
    tf_logger = configure(f"./tensorboard_logs/{model_name}_test", ["tensorboard"])
    model.set_logger(tf_logger)

    obs, _ = env.reset()
    done = False

    rewards = []
    # Initialize custom callback for logging with the environment passed in
    tensorboard_callback = TensorboardCallback(env, model)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        
        # Log reward and other metrics to TensorBoard using the callback
        tensorboard_callback.locals = {'rewards': reward}
        tensorboard_callback._on_step()  # Manually call _on_step() to log metrics
        
        rewards.append(reward)
    
    tf_logger.dump(step=0)  # Ensure logs are written

    metrics = [rewards, env.emissions_over_time, env.mean_speed_over_time, env.flows]
    result = {metric: value for metric, value in zip(metrics_to_plot, metrics)}

    return (model_name, result)

def test_trpo():
    model_name = "TRPO"
    model = TRPO.load(model_paths[model_name])
    # test_model(model, model_name)
    logging.debug(f"Starting {model_name} test")
    env = TrafficEnv(port=ports[model_name], model=model_name, model_idx=0)
    env.is_learning = False
    env.test_without_electric = test_without_electric
    env.test_without_disobedient = test_without_disobedient
    # check_env(env)
    
    # Set up TensorBoard logger
    tf_logger = configure(f"./tensorboard_logs/{model_name}_test", ["tensorboard"])
    model.set_logger(tf_logger)

    obs, _ = env.reset()
    done = False

    rewards = []
    # Initialize custom callback for logging with the environment passed in
    tensorboard_callback = TensorboardCallback(env, model)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        
        # Log reward and other metrics to TensorBoard using the callback
        tensorboard_callback.locals = {'rewards': reward}
        tensorboard_callback._on_step()  # Manually call _on_step() to log metrics
        
        rewards.append(reward)
    
    tf_logger.dump(step=0)  # Ensure logs are written

    metrics = [rewards, env.emissions_over_time, env.mean_speed_over_time, env.flows]
    result = {metric: value for metric, value in zip(metrics_to_plot, metrics)}

    return (model_name, result)

def test_sac():
    model_name = "SAC"
    model = SAC.load(model_paths[model_name])
    # test_model(model, model_name)
    logging.debug(f"Starting {model_name} test")
    env = TrafficEnv(port=ports[model_name], model=model_name, model_idx=0)
    env.is_learning = False
    env.test_without_electric = test_without_electric
    env.test_without_disobedient = test_without_disobedient
    # check_env(env)
    
    # Set up TensorBoard logger
    tf_logger = configure(f"./tensorboard_logs/{model_name}_test", ["tensorboard"])
    model.set_logger(tf_logger)

    obs, _ = env.reset()
    done = False

    rewards = []
    # Initialize custom callback for logging with the environment passed in
    tensorboard_callback = TensorboardCallback(env, model)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        
        # Log reward and other metrics to TensorBoard using the callback
        tensorboard_callback.locals = {'rewards': reward}
        tensorboard_callback._on_step()  # Manually call _on_step() to log metrics
        
        rewards.append(reward)
    
    tf_logger.dump(step=0)  # Ensure logs are written

    metrics = [rewards, env.emissions_over_time, env.mean_speed_over_time, env.flows]
    result = {metric: value for metric, value in zip(metrics_to_plot, metrics)}

    return (model_name, result)

def process_callback(result):
    agent_name, agent_result = result
    logging.debug(f"Processing result for {agent_name}")
    results[agent_name] = agent_result
    save_metrics(agent_result, agent_name)

def plot_metrics(selected_models=None):
    # Default to all models if none are specified
    if selected_models is None:
        selected_models = list(results.keys())

    # Calculate mean values
    mean_values = {agent: {metric: np.mean(data[metric]) for metric in data} for agent, data in results.items()}

    # Calculate variance and standard deviation
    variance_std_dev = {agent: {key: (np.var(values), np.std(values)) for key, values in metrics.items()} for agent, metrics in results.items()}

    """ Perform ANOVA 
    The script performs statistical analysis on the results using ANOVA (Analysis of Variance) to compare the performance of different agents across various metrics.
    This helps determine if there are statistically significant differences between the models' performances.
    """
    anova_results = {}
    for metric in results['PPO'].keys():
        data_for_anova = [results[agent][metric] for agent in selected_models]
        anova_result = stats.f_oneway(*data_for_anova)
        f_statistic = anova_result.statistic.item() if np.ndim(anova_result.statistic) > 0 else anova_result.statistic
        p_value = anova_result.pvalue.item() if np.ndim(anova_result.pvalue) > 0 else anova_result.pvalue
        anova_results[metric] = (f_statistic, p_value)

    # Log ANOVA Results to a file
    os.makedirs('logs', exist_ok=True)
    with open('logs/anova_results.log', 'w') as f:
        for metric in anova_results:
            f_statistic, p_value = anova_results[metric]
            f.write(f"ANOVA result for {metric}: F-statistic={f_statistic:.3f}, p-value={p_value:.3f}\n")

    # Calculate mean values and performance percentages
    mean_values = {agent: {metric: np.mean(data[metric]) for metric in data} for agent, data in results.items()}
    performance_percentage = {}

    for metric in metrics_to_plot:
        best_value = max(mean_values[agent][metric] for agent in mean_values)
        performance_percentage[metric] = {
            agent: ((best_value - mean_values[agent][metric]) / best_value) * 100
            for agent in mean_values
            }

    # Enable interactive mode
    # plt.ion()

    # Create figure with subplots
    plt.figure(figsize=(15, 10))

    # Plot each metric
    for i, metric in enumerate(metrics_to_plot):
        # Line plot for each metric
        plt.subplot(5, 2, i * 2 + 1)
        for agent_name in selected_models:
            _, results[agent_name]["rewards"] = reward_filter_flat_lines(results[agent_name]["rewards"])

            plt.plot(results[agent_name][metric],
                     label=f"{agent_name} ({performance_percentage[metric][agent_name]:.2f}%)",
                     color=colors.get(agent_name, 'black'))  # Default color if not found
        plt.xlabel("Iteration")
        plt.ylabel(metric.replace('_', ' ').capitalize())
        plt.title(f"{metric.replace('_', ' ').capitalize()} Comparison")
        plt.legend()

        # Bar plot for variance and standard deviation
        plt.subplot(5, 2, i * 2 + 2)
        std_devs = [variance_std_dev[agent][metric][1] for agent in selected_models]
        variances = [variance_std_dev[agent][metric][0] for agent in selected_models]

        bar_width = 0.35
        index = np.arange(len(selected_models))

        bars1 = plt.bar(index, std_devs, bar_width, label='Standard Deviation', color='deepskyblue')
        bars2 = plt.bar(index + bar_width, variances, bar_width, label='Variance', color='tomato')

        plt.xlabel('Agent')
        plt.ylabel('Value')
        plt.title(f'{metric.replace("_", " ").capitalize()} Variance and Standard Deviation')
        plt.xticks(index + bar_width / 2, selected_models)

        # Annotate bars with scaled values and adjusted position
        for bar in bars1:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2 - 0.1, yval + yval * 0.05, f'{yval:.1e}', fontsize=8)

        for bar in bars2:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2 - 0.1, yval + yval * 0.05, f'{yval:.1e}', fontsize=8)

        plt.legend()

    plt.tight_layout()
    plt.show(block=True)

if __name__ == '__main__':
    models_no = 6
    # flow_generation_wrapper(np.random.triangular(0.5, 1, 1.5), model="all", idx=0)
    test_without_electric = False
    test_without_disobedient = True
    sleep(1)

    # Ensure freeze_support() is called if necessary (typically for Windows)
    multiprocessing.freeze_support()

    logging.debug("Creating pool of processes")
    # Create a pool of processes
    pool = multiprocessing.Pool(processes=models_no)

    # Collect async results
    async_results = [
        pool.apply_async(test_dqn, callback=process_callback),
        pool.apply_async(test_ppo, callback=process_callback),
        pool.apply_async(test_a2c, callback=process_callback),
        pool.apply_async(test_trpo, callback=process_callback),
        pool.apply_async(test_sac, callback=process_callback)
        # pool.apply_async(test_td3, callback=process_callback)
    ]

    # Close the pool and wait for all processes to finish
    logging.debug("Closing pool")
    pool.close()
    pool.join()
    
    # Check if all async tasks were successful
    logging.debug("Checking async results")
    for async_result in async_results:
        try:
            async_result.get(timeout=10)
        except Exception as e:
            logging.error(f"An error occurred in one of the processes: {e}")

    # Once all processes are complete, plot the metrics
    logging.debug("Plotting metrics")
    plot_metrics()

    try:
        traci_close()
    except (FatalTraCIError, TraCIException):
        logging.debug(f"SUMO is not closing.")
