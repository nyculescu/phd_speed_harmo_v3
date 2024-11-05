import torch as th
from stable_baselines3 import PPO
import numpy as np

class MaxEntropyPPO(PPO):
    def __init__(self, *args, entropy_coef=0.01, **kwargs):
        super(MaxEntropyPPO, self).__init__(*args, **kwargs)
        self.entropy_coef = entropy_coef  # Coefficient for entropy regularization

    def train(self):
        # Call parent class's train method
        super().train()

        # Add maximum entropy term to loss
        for rollout_data in self.rollout_buffer.get(batch_size=self.batch_size):
            actions = rollout_data.actions
            log_probs = self.policy.get_distribution(rollout_data.observations).log_prob(actions)
            entropy_loss = -self.entropy_coef * log_probs.mean()  # Maximize entropy
            
            # Update policy loss with entropy term
            self.policy.optimizer.zero_grad()
            loss = self.policy_loss + entropy_loss
            loss.backward()
            self.policy.optimizer.step()

class DMEPPOLMI(MaxEntropyPPO):
    def __init__(self, *args, lmi_constraint_fn=None, lmi_penalty=1000.0, **kwargs):
        super(DMEPPOLMI, self).__init__(*args, **kwargs)
        self.lmi_constraint_fn = lmi_constraint_fn  # Function to check LMI constraints
        self.lmi_penalty = lmi_penalty  # Penalty for violating LMI constraints

    def train(self):
        # Call parent class's train method
        super().train()

        # After each update, check if LMI constraints are violated
        for rollout_data in self.rollout_buffer.get(batch_size=self.batch_size):
            actions = rollout_data.actions
            
            # Check if current policy violates LMI constraints
            if self.lmi_constraint_fn is not None:
                lmi_violated = self.lmi_constraint_fn(rollout_data.observations, actions)
                
                if lmi_violated:
                    # Apply penalty if LMI constraints are violated
                    penalty_loss = self.lmi_penalty * th.tensor(lmi_violated).float().mean()
                    
                    # Update policy loss with penalty term
                    self.policy.optimizer.zero_grad()
                    loss = penalty_loss + self.policy_loss  # Add penalty term
                    loss.backward()
                    self.policy.optimizer.step()

# Example usage:
def example_lmi_constraint_fn(observations, actions):
    # Dummy constraint function: returns True if actions violate some condition
    return np.any(actions > 1.0)  # Example condition

# env = (TrafficEnv(port=8880, model=model, model_idx=model_idx, is_learning=is_learning))

# Instantiate custom DMEPPOLMI model with LMI constraints
# model = DMEPPOLMI("MlpPolicy", env,
#                   lmi_constraint_fn=example_lmi_constraint_fn,
#                   verbose=1)
