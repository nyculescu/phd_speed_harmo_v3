import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.dqn.policies import DQNPolicy
import copy

class FPWDDQNReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super(FPWDDQNReplayBuffer, self).__init__(*args, **kwargs)
        # Initialize prioritization parameters here

    def add(self, *args, **kwargs):
        # Override to implement forgetful prioritization
        super().add(*args, **kwargs)
        # Adjust priorities and forget old samples

    def sample(self, batch_size, env=None):
        # Override to sample based on priority weights
        return super().sample(batch_size, env)

class FPWDDQN(DQN):
    def __init__(self, *args, **kwargs):
        super(FPWDDQN, self).__init__(*args, **kwargs)
        self.replay_buffer = FPWDDQNReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            optimize_memory_usage=self.optimize_memory_usage,
            n_envs=self.n_envs
        )
        # Initialize a second Q-network for Double DQN
        self.q_net_target2 = copy.deepcopy(self.policy.q_net)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        for _ in range(gradient_steps):
            # Sample a batch from replay buffer
            replay_data = self.replay_buffer.sample(batch_size)

            # Compute target Q-values using Double DQN logic
            with th.no_grad():
                next_actions = self.policy.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                q_values_next = self.q_net_target2(replay_data.next_observations)
                next_q_values = q_values_next.gather(1, next_actions.long()).squeeze(1)
                
                # Compute target values (shape [batch_size])
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values from main network (shape [batch_size])
            current_q_values = self.q_net(replay_data.observations).gather(1, replay_data.actions.long()).squeeze(-1)

            # Ensure shapes match before computing loss
            assert current_q_values.shape == target_q_values.shape, f"Shape mismatch: {current_q_values.shape} vs {target_q_values.shape}"

            # Compute loss (MSE between current and target Q-values)
            loss = th.nn.functional.mse_loss(current_q_values, target_q_values)

            # Optimize policy network
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()

    def _on_step(self):
        # Update target networks periodically
        if self.num_timesteps % self.target_update_interval == 0:
            self.q_net_target.load_state_dict(self.q_net.state_dict())
            self.q_net_target2.load_state_dict(self.q_net.state_dict())

#FIXME: 
# d:\phd_ws\speed_harmo\phd_speed_harmo_v3\rl_models\custom_models\DQN_FPWDDQN.py:50: UserWarning: Using a target size (torch.Size([32, 32])) that is different to the input size (torch.Size([32])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
#  loss = th.nn.functional.mse_loss(current_q_values, target_q_values)