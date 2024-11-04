import numpy as np

""" General configuration """
models = ["DQN", "A2C", "PPO", "TD3", "TRPO", "SAC", "DDPG"]
cont_act_space_models = ["TD3", "SAC", "DDPG"] # Models with continuous action spaces
discrete_act_space_models = ["DQN", "A2C", "PPO", "TRPO"] # Models with discrete action spaces

base_sumo_port = 8800
num_envs_per_model = 10 # it will replace the episodes, because through this, the episodes could be parallelized
test_without_electric = False
test_without_disobedient = False

model_paths = {
    "PPO": "rl_models/PPO/best_model",
    "A2C": "rl_models/A2C/best_model",
    "DQN": "rl_models/DQN/best_model",
    "TRPO": "rl_models/TRPO/best_model",
    "TD3": "rl_models/TD3/best_model",
    "SAC": "rl_models/SAC/best_model",
    "DDPG": "rl_models/DDPG/best_model"
}

""" Configuration for the model testing """
# Define colors for each agent
colors = {
    'PPO': 'grey',
    'A2C': 'violet',
    'DQN': 'turquoise',
    'TRPO': 'lightgreen',
    'TD3': 'khaki',
    'SAC': 'chocolate'
}
ports = {'PPO': 8810, 'A2C': 8811, 'DQN': 8812, 'TRPO': 8813, 'TD3': 8814, 'SAC': 8815}

results = {}

metrics_to_plot = ['rewards'
                #    , 'obs'
                   , 'emissions'
                   , 'mean speed'
                   , 'flow'
                   ]

"""  Configuration for the flow generation """
# Vehicle generation rates (bimodal distribution pattern)
daily_pattern = [
    np.random.randint(25,     75),   np.random.randint(50,     75), # 00:00-00:30-01:00
    np.random.randint(25,     50),   np.random.randint(10,     25), # 01:00-01:30-02:00
    np.random.randint(25,     50),   np.random.randint(5,      10), # 02:00-02:30-03:00
    np.random.randint(5,      25),   np.random.randint(10,     25), # 03:00-03:30-04:00 
    np.random.randint(25,     50),   np.random.randint(50,     75), # 04:00-04:30-05:00
    np.random.randint(50,    100),   np.random.randint(100,   150), # 05:00-05:30-06:00
    np.random.randint(200,   500),   np.random.randint(250,   500), # 06:00-06:30-07:00
    np.random.randint(750,  1000),   np.random.randint(1000, 1250), # 07:00-07:30-08:00
    np.random.randint(1250, 1500),   np.random.randint(1250, 1500), # 08:00-08:30-09:00
    np.random.randint(1250, 1500),   np.random.randint(1250, 1500), # 09:00-09:30-10:00
    np.random.randint(1000, 1250),   np.random.randint(1000, 1250), # 10:00-10:30-11:00
    np.random.randint(1000, 1250),   np.random.randint(1000, 1500), # 11:00-11:30-12:00
    np.random.randint(1250, 1500),   np.random.randint(1250, 1500), # 12:00-12:30-13:00
    np.random.randint(1250, 1750),   np.random.randint(1500, 1750), # 13:00-13:30-14:00
    np.random.randint(1250, 1500),   np.random.randint(1250, 1500), # 14:00-14:30-15:00
    np.random.randint(1250, 1500),   np.random.randint(1250, 1500), # 15:00-15:30-16:00
    np.random.randint(1250, 1500),   np.random.randint(1250, 1500), # 16:00-16:30-17:00
    np.random.randint(1500, 2000),   np.random.randint(1500, 1750), # 17:00-17:30-18:00
    np.random.randint(1500, 1750),   np.random.randint(1250, 1750), # 18:00-18:30-19:00
    np.random.randint(1000, 1250),   np.random.randint(1000, 1250), # 19:00-19:30-20:00
    np.random.randint(750,  1000),   np.random.randint(750,  1000), # 20:00-20:30-21:00
    np.random.randint(500,   750),   np.random.randint(500,   750), # 21:00-21:30-22:00
    np.random.randint(250,   500),   np.random.randint(100,   250), # 22:00-22:30-23:00
    np.random.randint(50,    250),   np.random.randint(50 ,   100)  # 23:00-23:30-00:00
]

# Day of the week factor # TODO: add this one in flow generation
day_of_the_week_factor = [
    np.random.triangular(0.95, 1, 1.05), # Monday
    np.random.triangular(0.90, 1, 1.10), # Tuesday
    np.random.triangular(0.90, 1, 1.10), # Wednesday
    np.random.triangular(0.90, 1, 1.10), # Thursday
    np.random.triangular(0.95, 1, 1.05), # Friday
    np.random.triangular(0.80, 1, 1.2),  # Saturday
    np.random.triangular(0.80, 1, 1.2)   # Sunday
]

car_generation_rates = len(daily_pattern) / 2 * len(day_of_the_week_factor)
addDisobedientVehicles = True
addElectricVehicles = True

# Day off factor (assuming no day off effect) # TODO: add this one in flow generation
day_off_factor = [
    1.0,  # Monday
    1.05, # Tuesday
    1.05, # Wednesday
    1.0,  # Thursday
    1.10, # Friday
    0.75, # Saturday
    0.8   # Sunday
]