import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DDPG
from stable_baselines3.common.policies import BasePolicy

class ConvLSTMLayer(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMLayer, self).__init__()
        self.conv_lstm = nn.LSTM(input_size=input_channels,
                                 hidden_size=hidden_channels,
                                 kernel_size=kernel_size,
                                 batch_first=True)

    def forward(self, x):
        # Assuming x is of shape [batch_size, seq_len, channels, height, width]
        batch_size, seq_len, channels, height, width = x.size()
        x = x.view(batch_size * seq_len, channels, height * width)
        output, _ = self.conv_lstm(x)
        return output.view(batch_size, seq_len, -1)

class LSTMActorCritic(nn.Module):
    def __init__(self):
        super(LSTMActorCritic, self).__init__()
        
        # Define a simple Conv layer followed by ConvLSTM
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv_lstm = ConvLSTMLayer(input_channels=16 * 26 * 26,
                                       hidden_channels=128,
                                       kernel_size=(3,))
        
        # Actor network
        self.actor_fc = nn.Linear(128, 1)  # Output action

        # Critic network
        self.critic_fc = nn.Linear(128 + 1, 1)  # Input state + action

    def forward_actor(self, x):
        # Forward pass through conv + convlstm for actor
        x = F.relu(self.conv(x))
        x = self.conv_lstm(x)
        action = torch.tanh(self.actor_fc(x))
        return action

    def forward_critic(self, x, action):
        # Forward pass through conv + convlstm for critic
        x = F.relu(self.conv(x))
        x = self.conv_lstm(x)
        q_value = self.critic_fc(torch.cat([x.flatten(), action], dim=-1))
        return q_value
    
class LSTMDDPG(DDPG):
    def __init__(self, *args, **kwargs):
        super(LSTMDDPG, self).__init__(*args, **kwargs)
        
        # Replace actor and critic networks with custom ones
        self.actor = LSTMActorCritic().forward_actor
        self.critic = LSTMActorCritic().forward_critic
