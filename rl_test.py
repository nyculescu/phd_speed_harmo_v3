import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_checker import check_env
from scipy import stats
import logging
import os
import multiprocessing
from time import sleep
from simulation_utilities.flow_gen import flow_generation

from rl_gym_environments import SUMOEnv

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
    "DQN": "rl_models/DQN/best_model"
}

# Define colors for each agent
colors = {'PPO': 'blue', 'A2C': 'orange', 'DQN': 'green'}

results = {}

metrics_to_plot = ['speed'
                   , 'emissions'
                   , 'halts_no'
                   , 'occupancy_curr'
                   , 'flow'
                   ]

def save_metrics(metrics, agent_name):
    with open(f'./logs/{agent_name}_metrics.csv', 'w+') as metrics_file:
        list_to_string = lambda x: ','.join([str(elem) for elem in x]) + '\n'
        metrics_file.write(list_to_string(metrics['speed']))
        metrics_file.write(list_to_string(metrics['flow']))
        metrics_file.write(list_to_string(metrics['emissions']))
        metrics_file.write(list_to_string(metrics['halts_no']))
        metrics_file.write(list_to_string(metrics['occupancy_curr']))
    
    # pd.DataFrame(metrics['cvs_seg_time']).to_csv(f'./metrics/{agent_name}.csv', index=False, header=False)

def test_ppo():
    logging.debug("Starting PPO test")
    ppo_env = SUMOEnv(port=8813)
    # check_env(ppo_env)
    ppo_model = PPO.load(model_paths['PPO'])

    obs, _ = ppo_env.reset()
    done = False

    while not done:
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = ppo_env.step(action)

    result = {metrics_to_plot[0]: ppo_env.mean_speeds_mps
              , metrics_to_plot[1]: ppo_env.mean_emissions
              , metrics_to_plot[2]: ppo_env.mean_num_halts
              , metrics_to_plot[3]: ppo_env.mean_occupancy_curr
              , metrics_to_plot[4]: ppo_env.flows
              }
    return ('PPO', result)

def test_a2c():
    logging.debug("Starting A2C test")
    a2c_env = SUMOEnv(port=8814)
    # check_env(a2c_env)
    a2c_model = A2C.load(model_paths['A2C'])
    
    obs, _ = a2c_env.reset()
    done = False

    while not done:
        action, _ = a2c_model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = a2c_env.step(action)
    
    result = {metrics_to_plot[0]: a2c_env.mean_speeds_mps
              , metrics_to_plot[1]: a2c_env.mean_emissions
              , metrics_to_plot[2]: a2c_env.mean_num_halts
              , metrics_to_plot[3]: a2c_env.mean_occupancy_curr
              , metrics_to_plot[4]: a2c_env.flows
              }
    
    return ('A2C', result)

# Note: even though the code seem to be redundant and can be abstractised into one function with different values of the params, leave it like this because it won't run on multi-process
def test_dqn():
    logging.debug("Starting DQN test")
    dqn_env = SUMOEnv(port=8815)
    # check_env(dqn_env)
    dqn_model = DQN.load(model_paths['DQN'])

    obs, _ = dqn_env.reset()
    done = False

    while not done:
        action, _ = dqn_model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = dqn_env.step(action)

    result = {metrics_to_plot[0]: dqn_env.mean_speeds_mps
              , metrics_to_plot[1]: dqn_env.mean_emissions
              , metrics_to_plot[2]: dqn_env.mean_num_halts
              , metrics_to_plot[3]: dqn_env.mean_occupancy_curr
              , metrics_to_plot[4]: dqn_env.flows
              }
    return ('DQN', result)

def process_callback(result):
    agent_name, agent_result = result
    logging.debug(f"Processing result for {agent_name}")
    results[agent_name] = agent_result
    save_metrics(agent_result, agent_name)

def plot_metrics():
    # Calculate mean values
    mean_values = {agent: {metric: np.mean(data[metric]) for metric in data} for agent, data in results.items()}

    # Calculate variance and standard deviation
    variance_std_dev = {agent: {key: (np.var(values), np.std(values)) for key, values in metrics.items()} for agent, metrics in results.items()}

    # Perform ANOVA
    anova_results = {
        'flow': stats.f_oneway(results['PPO']['flow'], results['A2C']['flow'], results['DQN']['flow']),
        'speed': stats.f_oneway(results['PPO']['speed'], results['A2C']['speed'], results['DQN']['speed']),
        'emissions': stats.f_oneway(results['PPO']['emissions'], results['A2C']['emissions'], results['DQN']['emissions']),
        'halts_no': stats.f_oneway(results['PPO']['halts_no'], results['A2C']['halts_no'], results['DQN']['halts_no']),
        'occupancy_curr': stats.f_oneway(results['PPO']['occupancy_curr'], results['A2C']['occupancy_curr'], results['DQN']['occupancy_curr']),
    }
    # Log ANOVA Results to a file
    os.makedirs('logs', exist_ok=True)
    with open('logs/anova_results.log', 'w') as f:
        for metric in anova_results:
            f.write(f"ANOVA result for {metric}: F-statistic={anova_results[metric].statistic:.3f}, p-value={anova_results[metric].pvalue:.3f}\n")

    def find_convergence_iterations(results):
        convergence = {}
        for agent, metrics in results.items():
            convergence[agent] = {key: np.argmax(values) + 1 for key, values in metrics.items()}
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

    # Create figure with subplots
    plt.figure(figsize=(15, 10))

    # Plot each metric
    for i, metric in enumerate(metrics_to_plot):
        # Line plot for each metric
        plt.subplot(5, 2, i * 2 + 1)
        for agent_name in results.keys():
            plt.plot(results[agent_name][metric],
                    label=f"{agent_name} ({performance_percentage[metric][agent_name]:.2f}%)",
                    color=colors[agent_name])
        plt.xlabel("Iteration")
        plt.ylabel(metric.replace('_', ' ').capitalize())
        plt.title(f"{metric.replace('_', ' ').capitalize()} Comparison")
        plt.legend()

        # Bar plot for variance and std dev
        plt.subplot(5, 2, i * 2 + 2)
        std_devs = [variance_std_dev[agent][metric][1] for agent in model_paths.keys()]
        variances = [variance_std_dev[agent][metric][0] for agent in model_paths.keys()]

        bar_width = 0.35
        index = np.arange(len(model_paths))

        bars1 = plt.bar(index, std_devs, bar_width, label='Std Dev', color='b')
        bars2 = plt.bar(index + bar_width, variances, bar_width, label='Variance', color='r')

        plt.xlabel('Agent')
        plt.ylabel('Value')
        plt.title(f'{metric.replace("_", " ").capitalize()} Variance and Std Dev')
        plt.xticks(index + bar_width / 2, model_paths.keys())

        # Annotate bars with scaled values and adjusted position
        for bar in bars1:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2 - 0.1, yval + yval * 0.05, f'{yval:.1e}', fontsize=8)

        for bar in bars2:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2 - 0.1, yval + yval * 0.05, f'{yval:.1e}', fontsize=8)

        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    flow_generation(np.random.triangular(0.5, 1, 1.5), day_index=0)
    sleep(1)

    # # Ensure freeze_support() is called if necessary (typically for Windows)
    # multiprocessing.freeze_support()

    # logging.debug("Creating pool of processes")
    # # Create a pool of processes
    # pool = multiprocessing.Pool(processes=3)

    # # Collect async results
    # async_results = [
    #     pool.apply_async(test_dqn, callback=process_callback),
    #     pool.apply_async(test_ppo, callback=process_callback),
    #     pool.apply_async(test_a2c, callback=process_callback)
    # ]

    # # Close the pool and wait for all processes to finish
    # logging.debug("Closing pool")
    # pool.close()
    # pool.join()
    
    # # Check if all async tasks were successful
    # logging.debug("Checking async results")
    # for async_result in async_results:
    #     try:
    #         async_result.get(timeout=10)
    #     except Exception as e:
    #         logging.error(f"An error occurred in one of the processes: {e}")

    # # FIXME: this one is called only for debugging purposes
    agent_name, agent_result = test_dqn()
    logging.debug(f"Processing result for {agent_name}")
    results[agent_name] = agent_result
    save_metrics(agent_result, agent_name)

    agent_name, agent_result = test_a2c()
    logging.debug(f"Processing result for {agent_name}")
    results[agent_name] = agent_result
    save_metrics(agent_result, agent_name)

    agent_name, agent_result = test_ppo()
    logging.debug(f"Processing result for {agent_name}")
    results[agent_name] = agent_result
    save_metrics(agent_result, agent_name)

    # Once all processes are complete, plot the metrics
    logging.debug("Plotting metrics")
    plot_metrics()
