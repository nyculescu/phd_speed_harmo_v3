import numpy as np
import os
import sys

""" General configuration """
cont_act_space_models = ["TD3", "SAC", "DDPG"] # Models with continuous action spaces
discrete_act_space_models = ["TRPO", "DQN", "A2C", "PPO"] # Models with discrete action spaces
all_models = cont_act_space_models + discrete_act_space_models

base_sumo_port = 8800
num_envs_per_model = 10 # it will replace the episodes, because through this, the episodes could be parallelized
test_with_electric = True # default value
test_with_disobedient = True # default value

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
    'SAC': 'chocolate',
    'DDPG': 'darkblue'
}
ports = {'PPO': 8810, 'A2C': 8811, 'DQN': 8812, 'TRPO': 8813, 'TD3': 8814, 'SAC': 8815, 'DDPG': 8816}

results = {}

metrics_to_plot = ['rewards'
                #    , 'obs'
                   , 'emissions'
                   , 'mean speed'
                   , 'flow'
                   ]

"""  Configuration for the flow generation """
mock_daily_pattern = [
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
    np.random.uniform(0.95, 1.05), # Monday
    np.random.uniform(0.90, 1.10), # Tuesday
    np.random.uniform(0.90, 1.10), # Wednesday
    np.random.uniform(0.90, 1.10), # Thursday
    np.random.uniform(0.95, 1.05), # Friday
    np.random.uniform(0.80, 1.2),  # Saturday
    np.random.uniform(0.80, 1.2)   # Sunday
]

mock_days_and_weeks = False
base_demand = np.random.randint(1300, 1800) # max no. of vehicles expected in any interval
car_generation_rates = len(mock_daily_pattern) / 2 * len(day_of_the_week_factor)
addDisobedientVehicles = True
addElectricVehicles = True

# Day off factor (assuming no day off effect) # TODO: add this one in flow generation
day_off_factor = [
    np.random.uniform(0.95, 1.05),  # Monday
    np.random.uniform(1.0, 1.05), # Tuesday
    np.random.uniform(1.0, 1.05), # Wednesday
    np.random.uniform(0.95, 1.05),  # Thursday
    np.random.uniform(1.0, 1.10), # Friday
    np.random.uniform(0.7, 0.8), # Saturday
    np.random.uniform(0.7, 0.8)   # Sunday
]

""" Configuration for the environment """
# from https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html
def check_sumo_env():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
# Notes: 
#     A smaller step length (e.g., 0.5) => more frequent updates, which can slow down the simulation but increase accuracy. Conversely, increasing it (e.g., 1.0 or more) will speed up the simulation at the cost of some detail.  '--step-length', '0.5'
# Running SUMO without the graphical interface can significantly speed up the simulation
# sumoExecutable = 'sumo.exe' if os.name == 'nt' else 'sumo'
# -d, --delay FLOAT  Use FLOAT in ms as delay between simulation steps
sumoExecutable = 'sumo-gui.exe' if os.name == 'nt' else 'sumo-gui'
sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', sumoExecutable)
