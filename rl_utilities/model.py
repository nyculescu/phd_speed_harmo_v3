import tensorflow as tf

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_model(states, actions):
    model = tf.keras.models.Sequential()    
    model.add(tf.keras.layers.Dense(24, activation='relu', input_shape=states))
    model.add(tf.keras.layers.Dense(24, activation='relu'))
    model.add(tf.keras.layers.Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn