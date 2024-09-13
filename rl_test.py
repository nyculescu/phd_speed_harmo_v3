from matplotlib import pyplot as plt
import pandas as pd

from rl_gym_environments import *
from rl_utilities.model import *

env = SUMOEnv()
actions = env.action_space.n
states = env.observation_space.shape
model = build_model(states, actions)
dqn = build_agent(model, actions)
dqn.compile(tf.keras.optimizers.Adam(lr=1e-3), metrics=['mae'])

dqn.load_weights('rl_models/dqn_occ_weights.h5f')

_ = dqn.test(env, nb_episodes=1, visualize=False)

# Save metrics into csv file.
approach = 'rl'
with open(f'./metrics/{approach}_metrics.csv', 'w+') as metrics_file:
    list_to_string = lambda x: ','.join([ str(elem) for elem in x ]) + '\n'
    metrics_file.write(list_to_string(env.mean_speeds))
    metrics_file.write(list_to_string(env.flows))
    metrics_file.write(list_to_string(env.emissions_over_time))

pd.DataFrame(env.cvs_seg_time).to_csv(f'./metrics/{approach}_cvs.csv', index=False, header=False)

# plot occupancy and flow diagram to get capacity flow    
# fig, ax = plt.subplots(1,1, figsize=(15,30)) 
# plt.xticks(np.arange(min(env.state), max(env.state)+1, 1.0))
# plt.xlabel("Occupancy [%]")
# plt.ylabel("Flow [veh/h]")
# plt.title("")
# plt.plot(env.state, env.flows, 'bo')
# plt.show() 

plt.xlabel("Iteration")
plt.ylabel("Flow")
plt.title("Flow")
plt.plot(env.flows)
plt.show()

plt.xlabel("Iteration")
plt.ylabel("Mean speed")
plt.title("Mean speed over the whole stretch")
plt.plot(env.mean_speeds)
plt.show()

plt.xlabel("Step")
plt.ylabel("Emission level")
plt.title("Emissions over time")
plt.plot(env.emissions_over_time)
plt.show()