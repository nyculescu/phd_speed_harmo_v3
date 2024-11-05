import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.dqn.policies import DQNPolicy

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
        self.q_net_target2 = self.policy.q_net.clone()

    def train(self, gradient_steps: int, batch_size: int) -> None:
        for _ in range(gradient_steps):
            # Sample a batch
            replay_data = self.replay_buffer.sample(batch_size)
            # Compute target using Double DQN logic
            with th.no_grad():
                # Select actions using the first Q-network
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1)
                # Compute Q-values from the second Q-network
                next_q_values = self.q_net_target2(replay_data.next_observations).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
                # Compute targets
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
            # Get current Q-values
            current_q_values = self.q_net(replay_data.observations).gather(1, replay_data.actions.long()).squeeze(-1)
            # Compute loss
            loss = th.nn.functional.mse_loss(current_q_values, target_q_values)
            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()
            # Update replay buffer priorities if using prioritized replay

    def _on_step(self):
        # Update target networks periodically
        if self.num_timesteps % self.target_update_interval == 0:
            self.q_net_target.load_state_dict(self.q_net.state_dict())
            self.q_net_target2.load_state_dict(self.q_net.state_dict())