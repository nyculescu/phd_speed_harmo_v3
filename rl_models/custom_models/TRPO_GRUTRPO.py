import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

class GRUPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule):
        super(GRUPolicy, self).__init__(observation_space, action_space, lr_schedule)
        
        # Define a feature extractor (e.g., MLP)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Define a GRU layer
        self.gru = nn.GRU(input_size=64, hidden_size=32, batch_first=True)
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space.shape[0])
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, obs):
        # Extract features
        features = self.feature_extractor(obs)
        
        # Pass through GRU layer
        gru_output, _ = self.gru(features.unsqueeze(0))  # Add batch dimension for GRU
        
        # Get action and value from actor and critic networks
        action = self.actor(gru_output.squeeze(0))
        value = self.critic(gru_output.squeeze(0))
        
        return action, value
    
    def _predict(self, obs):
        action_logits = self.forward(obs)[0]
        return action_logits
