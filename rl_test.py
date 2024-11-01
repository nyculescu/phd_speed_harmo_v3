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

from traffic_environment.rl_gym_environments import TrafficEnv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the log level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[  # Handlers determine where logs are sent
        logging.StreamHandler()  # Output logs to stderr (default)
    ]
)

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
    logging.debug(f"Starting {model_name} test")
    env = TrafficEnv(port=ports[model_name], model=model_name, model_idx=0)
    env.is_learning = False
    # check_env(env)
    ppo_model = PPO.load(model_paths[model_name])

    obs, _ = env.reset()
    done = False

    rewards = []
    obss = []
    while not done:
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        obss.append(obs)
        rewards.append(reward)
    # obss = np.array(obss)

    metrics = [rewards, obss, env.emissions_over_time, env.mean_speed_over_time, env.flows]
    result = {metric: value for metric, value in zip(metrics_to_plot, metrics)}
    return (model_name, result)

def test_a2c():
    model_name = "A2C"
    logging.debug(f"Starting {model_name} test")
    env = TrafficEnv(port=ports[model_name], model=model_name, model_idx=0)
    env.is_learning = False
    # check_env(env)
    a2c_model = A2C.load(model_paths[model_name])
    
    obs, _ = env.reset()
    done = False

    rewards = []
    obss = []
    while not done:
        action, _ = a2c_model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        obss.append(obs)
        rewards.append(reward)
    
    metrics = [rewards, obss, env.emissions_over_time, env.mean_speed_over_time, env.flows]
    result = {metric: value for metric, value in zip(metrics_to_plot, metrics)}
    
    return (model_name, result)

# Note: redundant code on purpose, else it won't run on multi-process
def test_dqn():
    model_name = "DQN"
    logging.debug(f"Starting {model_name} test")
    env = TrafficEnv(port=ports[model_name], model=model_name, model_idx=0)
    env.is_learning = False
    # check_env(env)
    dqn_model = DQN.load(model_paths[model_name])

    obs, _ = env.reset()
    done = False

    rewards = []
    obss = []
    while not done:
        action, _ = dqn_model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        obss.append(obs)
        rewards.append(reward)
    
    metrics = [rewards, obss, env.emissions_over_time, env.mean_speed_over_time, env.flows]
    result = {metric: value for metric, value in zip(metrics_to_plot, metrics)}

    return (model_name, result)

def test_td3():
    model_name = "TD3"
    logging.debug(f"Starting {model_name} test")
    env = TrafficEnv(port=ports[model_name], model=model_name, model_idx=0)
    env.is_learning = False
    # check_env(dqn_env)
    model = TD3.load(model_paths[model_name])

    obs, _ = env.reset()
    done = False

    rewards = []
    obss = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        obss.append(obs)
        rewards.append(reward)
    
    metrics = [rewards, obss, env.emissions_over_time, env.mean_speed_over_time, env.flows]
    result = {metric: value for metric, value in zip(metrics_to_plot, metrics)}

    return (model_name, result)

def test_trpo():
    model_name = "TRPO"
    logging.debug(f"Starting {model_name} test")
    env = TrafficEnv(port=ports[model_name], model=model_name, model_idx=0)
    env.is_learning = False
    # check_env(dqn_env)
    model = TRPO.load(model_paths[model_name])

    obs, _ = env.reset()
    done = False

    rewards = []
    obss = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        obss.append(obs)
        rewards.append(reward)
    
    metrics = [rewards, obss, env.emissions_over_time, env.mean_speed_over_time, env.flows]
    result = {metric: value for metric, value in zip(metrics_to_plot, metrics)}

    return (model_name, result)

def test_sac():
    model_name = "SAC"
    logging.debug(f"Starting {model_name} test")
    env = TrafficEnv(port=ports[model_name], model=model_name, model_idx=0)
    env.is_learning = False
    # check_env(dqn_env)
    model = SAC.load(model_paths[model_name])

    obs, _ = env.reset()
    done = False

    rewards = []
    obss = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        obss.append(obs)
        rewards.append(reward)
    
    metrics = [rewards, obss, env.emissions_over_time, env.mean_speed_over_time, env.flows]
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

    # Perform ANOVA
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

    def find_convergence_iterations(results):
        convergence = {}
        for agent, metrics in results.items():
            convergence[agent] = {}
            for key, values in metrics.items():
                if len(values) > 0:
                    convergence[agent][key] = np.argmax(values) + 1
                else:
                    convergence[agent][key] = None  # Or some other default value indicating no data
        return convergence

    convergence_iterations = find_convergence_iterations(results)
    
    # Log convergence iterations
    with open('logs/convergence_iterations.log', 'w') as f:
        for agent in convergence_iterations:
            f.write(f"Convergence iterations for {agent}:\n")
            for metric, iteration in convergence_iterations[agent].items():
                f.write(f"  {metric}: {iteration}\n")

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
            plt.plot(results[agent_name][metric],
                     label=f"{agent_name} ({performance_percentage[metric][agent_name]:.2f}%)",
                     color=colors.get(agent_name, 'black'))  # Default color if not found
        plt.xlabel("Iteration")
        plt.ylabel(metric.replace('_', ' ').capitalize())
        plt.title(f"{metric.replace('_', ' ').capitalize()} Comparison")
        plt.legend()

        # Bar plot for variance and std dev
        plt.subplot(5, 2, i * 2 + 2)
        std_devs = [variance_std_dev[agent][metric][1] for agent in selected_models]
        variances = [variance_std_dev[agent][metric][0] for agent in selected_models]

        bar_width = 0.35
        index = np.arange(len(selected_models))

        bars1 = plt.bar(index, std_devs, bar_width, label='Std Dev', color='deepskyblue')
        bars2 = plt.bar(index + bar_width, variances, bar_width, label='Variance', color='tomato')

        plt.xlabel('Agent')
        plt.ylabel('Value')
        plt.title(f'{metric.replace("_", " ").capitalize()} Variance and Std Dev')
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
    flow_generation_wrapper(np.random.triangular(0.5, 1, 1.5), model="all", idx=0)
    # sleep(1)

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
        # pool.apply_async(test_td3, callback=process_callback),
        pool.apply_async(test_sac, callback=process_callback)
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

    # # FIXME: this one is called only for debugging purposes
    # flow_generation_wrapper(np.random.triangular(0.5, 1, 1.5), day_index=0, model="DQN", idx=0)
    # agent_name, agent_result = test_dqn()
    # logging.debug(f"Processing result for {agent_name}")
    # results[agent_name] = agent_result
    # save_metrics(agent_result, agent_name)

    # flow_generation_wrapper(np.random.triangular(0.5, 1, 1.5), day_index=0, model="A2C", idx=0)
    # agent_name, agent_result = test_a2c()
    # logging.debug(f"Processing result for {agent_name}")
    # results[agent_name] = agent_result
    # save_metrics(agent_result, agent_name)

    # flow_generation_wrapper(np.random.triangular(0.5, 1, 1.5), day_index=0, model="PPO", idx=0)
    # agent_name, agent_result = test_ppo()
    # logging.debug(f"Processing result for {agent_name}")
    # results[agent_name] = agent_result
    # save_metrics(agent_result, agent_name)

    # Once all processes are complete, plot the metrics
    logging.debug("Plotting metrics")
    # plot_metrics()
    plot_metrics()

    try:
        traci_close()
    except (FatalTraCIError, TraCIException):
        logging.debug(f"SUMO is not closing.")
