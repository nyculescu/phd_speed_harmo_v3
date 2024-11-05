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

    def add(self, obs, next_obs, action, reward, done):
        # Add new experience and set initial priority
        super().add(obs, next_obs, action, reward, done)
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0
        self.priorities[self.pos] = max_priority

    def sample(self, batch_size):
        # Compute probabilities based on priorities
        priorities = self.priorities[:self.size] ** self.alpha
        probabilities = priorities / np.sum(priorities)

        # Sample indices based on probability distribution
        indices = np.random.choice(self.size, batch_size, p=probabilities)

        # Importance-sampling weights
        total = self.size
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        # Increment beta towards 1 over time
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        # Get samples from replay buffer
        batch = super().sample(batch_size)
        
        # Add importance-sampling weights and indices to the batch
        batch['weights'] = weights
        batch['indices'] = indices

        return batch

    def update_priorities(self, indices, td_errors):
        # Update priorities based on TD-errors
        self.priorities[indices] = np.abs(td_errors) + 1e-5  # Add small epsilon to avoid zero priority

from stable_baselines3 import DDPG

class PRDDPG(DDPG):
    def __init__(self, *args, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, **kwargs):
        super(PRDDPG, self).__init__(*args, **kwargs)
        
        # Use custom prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
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
