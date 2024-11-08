import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer

class PRReplayBuffer(ReplayBuffer):
    def __init__(self, *args, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, **kwargs):
        super(PRReplayBuffer, self).__init__(*args, **kwargs)
        self.alpha = alpha  # Controls how much prioritization is used (0 = uniform)
        self.beta = beta  # Controls importance-sampling correction
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos=None):
        # Add new experience and set initial priority
        super().add(obs, next_obs, action, reward, done, infos)
        current_size = len(self)
        # Get valid priorities
        if current_size > 0:
            max_priority = np.max(self.priorities[:current_size])
        else:
            max_priority = 1.0  # Initial priority
        # Update priority at the correct index
        idx = (self.pos - 1) % self.buffer_size
        self.priorities[idx] = max_priority

    def sample(self, batch_size):
        current_size = len(self)
        # Compute probabilities based on priorities
        priorities = self.priorities[:current_size] ** self.alpha
        probabilities = priorities / priorities.sum()

        # Sample indices based on probability distribution
        indices = np.random.choice(current_size, batch_size, p=probabilities)

        # Importance-sampling weights
        total = current_size
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability

        # Get samples from the base class method
        data = super()._get_samples(indices)
        data = data._replace(weights=weights)
        return data

    def update_priorities(self, indices, td_errors):
        # Update priorities based on TD-errors
        self.priorities[indices] = (np.abs(td_errors) + 1e-5) ** self.alpha

    def __len__(self):
        return self.pos if not self.full else self.buffer_size

from stable_baselines3 import DDPG

class PRDDPG(DDPG):
    def __init__(self, *args, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, **kwargs):
        super(PRDDPG, self).__init__(*args, **kwargs)
        
        # Use custom prioritized replay buffer
        self.replay_buffer = PRReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            optimize_memory_usage=self.optimize_memory_usage,
            n_envs=self.n_envs,
            alpha=alpha,
            beta=beta,
            beta_increment_per_sampling=beta_increment_per_sampling
        )

    def train(self, gradient_steps: int, batch_size: int) -> None:
        for _ in range(gradient_steps):
            # Sample from prioritized replay buffer
            replay_data = self.replay_buffer.sample(batch_size)

            # Compute target Q-values using target networks (as in standard DDPG)
            with th.no_grad():
                next_actions = self.actor_target(replay_data.next_observations)
                next_q_values = self.critic_target(replay_data.next_observations, next_actions)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values from critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute TD-errors for prioritized experience replay update
            td_errors = target_q_values - current_q_values

            # Update critic network using importance-sampling weights
            loss_critic = (replay_data.weights * th.nn.functional.mse_loss(current_q_values.flatten(), target_q_values.flatten(), reduction='none')).mean()
            
            self.policy.optimizer.zero_grad()
            loss_critic.backward()
            self.policy.optimizer.step()

            # Update priorities in the replay buffer based on TD-errors
            self.replay_buffer.update_priorities(replay_data.indices.cpu().numpy(), td_errors.cpu().numpy())

# FIXME: ERROR:root:An error occurred in one of the processes: 'PRDDPG' object has no attribute 'observation_space'