import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_checker import check_env
from scipy import stats
import logging
import os
import multiprocessing

from rl_gym_environments import SUMOEnv

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

results = {
    "PPO": "",
    "A2C": "",
    "DQN": ""
}

def save_metrics(metrics, agent_name):
    with open(f'./metrics/{agent_name}_metrics.csv', 'w+') as metrics_file:
        list_to_string = lambda x: ','.join([str(elem) for elem in x]) + '\n'
        metrics_file.write(list_to_string(metrics['mean_speeds']))
        metrics_file.write(list_to_string(metrics['flows']))
        metrics_file.write(list_to_string(metrics['emissions_over_time']))
    
    pd.DataFrame(metrics['cvs_seg_time']).to_csv(f'./metrics/{agent_name}.csv', index=False, header=False)

def test_ppo():
    ppo_env = SUMOEnv(port=8813)
    check_env(ppo_env)
    ppo_model = PPO.load(model_paths['PPO'])

    obs, _ = ppo_env.reset()
    done = False

    while not done:
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = ppo_env.step(action)

    result = {'mean_speeds': ppo_env.mean_speeds,
        'flows': ppo_env.flows,
        'emissions_over_time': ppo_env.emissions_over_time,
        'cvs_seg_time': ppo_env.cvs_seg_time}
    return ('PPO', result)

def test_a2c():
    a2c_env = SUMOEnv(port=8814)
    check_env(a2c_env)
    a2c_model = A2C.load(model_paths['A2C'])
    
    obs, _ = a2c_env.reset()
    done = False

    while not done:
        action, _ = a2c_model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = a2c_env.step(action)

    result = {'mean_speeds': a2c_env.mean_speeds,
        'flows': a2c_env.flows,
        'emissions_over_time': a2c_env.emissions_over_time,
        'cvs_seg_time': a2c_env.cvs_seg_time}
    
    return ('A2C', result)

# Note: even though the code seem to be redundant and can be abstractised into one function with different values of the params, leave it like this because it won't run on multi-process
def test_dqn():
    dqn_env = SUMOEnv(port=8815)
    check_env(dqn_env)
    dqn_model = DQN.load(model_paths['DQN'])

    obs, _ = dqn_env.reset()
    done = False

    while not done:
        action, _ = dqn_model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = dqn_env.step(action)

    result = {'mean_speeds': dqn_env.mean_speeds,
        'flows': dqn_env.flows,
        'emissions_over_time': dqn_env.emissions_over_time,
        'cvs_seg_time': dqn_env.cvs_seg_time}
    return ('DQN', result)

def process_callback(result):
    agent_name, agent_result = result
    results[agent_name] = agent_result
    save_metrics(agent_result, agent_name)

def plot_metrics():
    # Calculate mean values
    mean_values = {agent: {metric: np.mean(data[metric]) for metric in data} for agent, data in results.items()}

    # Calculate performance percentages compared to PPO
    performance_percentage = {
        metric: {
            agent: (mean_values[agent][metric] - mean_values['PPO'][metric]) / mean_values['PPO'][metric] * 100
            for agent in ['A2C', 'DQN']
        }
        for metric in ['flows', 'mean_speeds', 'emissions_over_time']
    }

    # Calculate variance and standard deviation
    variance_std_dev = {agent: {key: (np.var(values), np.std(values)) for key, values in metrics.items()} for agent, metrics in results.items()}

    # Perform ANOVA
    anova_results = {
        'flows': stats.f_oneway(results['PPO']['flows'], results['A2C']['flows'], results['DQN']['flows']),
        'mean_speeds': stats.f_oneway(results['PPO']['mean_speeds'], results['A2C']['mean_speeds'], results['DQN']['mean_speeds']),
        'emissions': stats.f_oneway(results['PPO']['emissions_over_time'], results['A2C']['emissions_over_time'], results['DQN']['emissions_over_time'])
    }
    # Log ANOVA Results to a file
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/anova_results.log', 'w') as f:
        for metric in anova_results:
            f.write(f"ANOVA result for {metric}: F-statistic={anova_results[metric].statistic:.3f}, p-value={anova_results[metric].pvalue:.3f}\n")

    def find_convergence_iterations(results):
        convergence = {}
        for agent, metrics in results.items():
            convergence[agent] = {key: np.argmax(values) + 1 for key, values in metrics.items()}
        return convergence

    convergence_iterations = find_convergence_iterations(results)
    # Log convergence iterations
    with open('metrics/convergence_iterations.log', 'w') as f:
        for agent in convergence_iterations:
            f.write(f"Convergence iterations for {agent}:\n")
            for metric, iteration in convergence_iterations[agent].items():
                f.write(f"  {metric}: {iteration}\n")

    # Plots
    plt.figure(figsize=(15, 5))

    # Flow Comparison
    plt.subplot(231)
    for agent_name, data in results.items():
        plt.plot(data['flows'], label=f"{agent_name} ({performance_percentage['flows'].get(agent_name, 0):.2f}%)", color=colors[agent_name])
    plt.xlabel("Iteration")
    plt.ylabel("Flow")
    plt.title("Flow Comparison")
    plt.legend()

    # Mean Speed Comparison
    plt.subplot(232)
    for agent_name, data in results.items():
        plt.plot(data['mean_speeds'], label=f"{agent_name} ({performance_percentage['mean_speeds'].get(agent_name, 0):.2f}%)", color=colors[agent_name])
    plt.xlabel("Iteration")
    plt.ylabel("Mean Speed")
    plt.title("Mean Speed Comparison")
    plt.legend()

    # Emissions Comparison
    plt.subplot(233)
    for agent_name, data in results.items():
        plt.plot(data['emissions_over_time'], label=f"{agent_name} ({performance_percentage['emissions_over_time'].get(agent_name, 0):.2f}%)", color=colors[agent_name])
    plt.xlabel("Step")
    plt.ylabel("Emission Level")
    plt.title("Emissions Comparison")
    plt.legend()

    # Variance and Std Dev Plot with Annotations
    metrics_to_plot = ['flows', 'mean_speeds', 'emissions_over_time']
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(234 + i)
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
            plt.text(bar.get_x() + bar.get_width()/2.0 - 0.1, yval + yval * 0.05, f'{yval:.1e}', fontsize=8)
        
        for bar in bars2:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0 - 0.1, yval + yval * 0.05, f'{yval:.1e}', fontsize=8)
        
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Ensure freeze_support() is called if necessary (typically for Windows)
    multiprocessing.freeze_support()

    # Create a pool of processes
    pool = multiprocessing.Pool(processes=3)

    # Use apply_async to start each test function with a callback
    pool.apply_async(test_dqn, callback=process_callback)
    pool.apply_async(test_ppo, callback=process_callback)
    pool.apply_async(test_a2c, callback=process_callback)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    # Once all processes are complete, plot the metrics
    plot_metrics()