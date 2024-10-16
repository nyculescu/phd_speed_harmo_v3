import numpy as np
import matplotlib.pyplot as plt
from traffic_environment.rl_gym_environments import models

# for model in models:
model = 'DQN'
# Load evaluation data
data = np.load(f'./logs/{model}/evaluations.npz')
# Extract metrics
timesteps = data['timesteps']
results = data['results']  # Shape: (n_eval_episodes, n_metrics)
# Plot average reward over evaluations
mean_rewards = results.mean(axis=1)
plt.figure(num=f'{model} eval mean reward')
plt.plot(timesteps, mean_rewards)
plt.xlabel('Timesteps')
plt.ylabel('Mean Reward')
plt.title(f'Evaluation Mean Reward Over Time for the model {model}')
plt.show()