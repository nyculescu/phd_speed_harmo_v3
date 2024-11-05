from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
import torch as th

class STMReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super(STMReplayBuffer, self).__init__(*args, **kwargs)
        # Add an additional buffer for storing transition speeds
        self.transition_speeds = np.zeros((self.buffer_size,), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, speed):
        # Store transition speed along with other data
        super().add(obs, next_obs, action, reward, done)
        self.transition_speeds[self.pos] = speed

    def sample(self, batch_size):
        # Sample transitions as usual but include transition speeds
        batch = super().sample(batch_size)
        batch['speeds'] = self.transition_speeds[batch['indices']]
        return batch
    
class STMDQN(DQN):
    def __init__(self, *args, **kwargs):
        super(STMDQN, self).__init__(*args, **kwargs)
        # Use custom replay buffer with STM support
        self.replay_buffer = STMReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            optimize_memory_usage=self.optimize_memory_usage,
            n_envs=self.n_envs
        )

    def train(self, gradient_steps: int, batch_size: int) -> None:
        for _ in range(gradient_steps):
            # Sample a batch from replay buffer
            replay_data = self.replay_buffer.sample(batch_size)

            # Compute target Q-values using STM logic
            with th.no_grad():
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1)
                next_q_values = self.q_net_target(replay_data.next_observations).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)

                # Incorporate transition speeds into target calculation (e.g., penalize slow transitions)
                speed_penalty = 1 / (replay_data.speeds + 1e-5)  # Example penalty based on speed
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values * speed_penalty

            # Compute current Q-values
            current_q_values = self.q_net(replay_data.observations).gather(1, replay_data.actions.long()).squeeze(-1)

            # Compute loss and optimize model
            loss = th.nn.functional.mse_loss(current_q_values, target_q_values)
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()
    
    def _on_step(self):
        if self.num_timesteps % self.target_update_interval == 0:
            self.q_net_target.load_state_dict(self.q_net.state_dict())
