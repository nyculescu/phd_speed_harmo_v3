from stable_baselines3 import DQN
import torch

class DoubleDQN(DQN):
    def __init__(self, policy, env, **kwargs):
        super().__init__(policy=policy, env=env, **kwargs)
    
    def _target_q_value(self, replay_data):
        with torch.no_grad():
            # Select actions using online network
            next_q_values = self.q_net(replay_data.next_observations)
            next_actions = next_q_values.argmax(dim=1).reshape(-1, 1)
            # Evaluate actions using target network
            next_q_values_target = self.q_net_target(replay_data.next_observations)
            target_q_values = next_q_values_target.gather(1, next_actions)
        return target_q_values
