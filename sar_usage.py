# example_sar_usage.py
"""
Example usage of the modular SAR framework for DRL-VSL

This script demonstrates how to:
1. Use different SAR configurations
2. Create custom SAR components
3. Run experiments with different combinations
"""

import os
import sys
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# Import the refactored environment and SAR framework
from legacy.drl_vsl import create_sumocfg
from core.drl_vsl_refactored import TrafficEnv
from core.sar_framework import (
    StateRepresentation, ActionStrategy, RewardFunction,
    TrafficMetrics, create_state_representation,
    create_action_strategy, create_reward_function
)

# ============================================================================
# CUSTOM SAR COMPONENTS EXAMPLES
# ============================================================================

class CompactState(StateRepresentation):
    """Example custom state: Only flow and occupancy"""
    
    def _setup(self):
        self.num_features = 2
        self.max_flow = self.config.get('max_flow', 10000.0)
    
    def get_observation_space(self):
        import gymnasium as gym
        return gym.spaces.Box(
            low=np.zeros(self.num_features, dtype=np.float64),
            high=np.ones(self.num_features, dtype=np.float64),
            shape=(self.num_features,),
            dtype=np.float64
        )
    
    def build_state(self, metrics: TrafficMetrics):
        return np.array([
            metrics.flow_downstream,
            metrics.occupancy_upstream
        ], dtype=np.float64)
    
    def preprocess_state(self, raw_state):
        return np.array([
            np.clip(raw_state[0], 0, self.max_flow) / self.max_flow,
            np.clip(raw_state[1], 0, 100) / 100.0
        ], dtype=np.float64)


class ConservativeAction(ActionStrategy):
    """Example custom action: Small speed changes only"""
    
    def _setup(self):
        self.speed_changes = [-5, 0, 5]  # Only small changes
        self.current_speed = 130
    
    def get_action_space(self):
        import gymnasium as gym
        return gym.spaces.Discrete(len(self.speed_changes))
    
    def apply_action(self, action, current_speed_limit):
        change = self.speed_changes[action]
        new_speed = np.clip(current_speed_limit + change, 60, 130)
        penalty = -0.1 if change != 0 else 0  # Small penalty for any change
        return new_speed, penalty


class EmissionReward(RewardFunction):
    """Example custom reward: Focus on emissions reduction"""
    
    def calculate(self, metrics, action_penalty=0.0, collision_penalty=0.0):
        # Emissions are lowest at steady speeds around 90 km/h
        optimal_speed_ms = 90 / 3.6
        
        # Speed efficiency component
        if metrics.avg_speed_before > 0:
            speed_deviation = abs(metrics.avg_speed_before - optimal_speed_ms)
            speed_efficiency = np.exp(-speed_deviation / 10.0)
        else:
            speed_efficiency = 0
        
        # Flow component (still want good throughput)
        flow_component = min(metrics.flow_smoothed / self.max_flow, 1.0) * 0.3
        
        # Stop-and-go penalty (high variance = more emissions)
        if len(metrics.speed_history) > 5:
            speed_variance = np.var(list(metrics.speed_history))
            stability_component = np.exp(-speed_variance / 100.0) * 0.4
        else:
            stability_component = 0.2
        
        return float(speed_efficiency * 0.3 + flow_component + stability_component + 
                    action_penalty + collision_penalty)


# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

def create_env_with_sar(env_idx, port, model_name, sim_length, 
                        state_name, action_name, reward_name,
                        sar_config=None):
    """Factory function to create environment with specific SAR configuration"""
    
    # Create the SUMO config file BEFORE creating the environment
    effective_model_name = f"{model_name}_{env_idx}" if os.environ.get('OPTION') == '1' else model_name
    create_sumocfg(effective_model_name)
    
    # Create SAR components
    state_repr = create_state_representation(state_name, sar_config or {})
    action_strat = create_action_strategy(action_name, sar_config or {})
    reward_func = create_reward_function(reward_name, sar_config or {})
    
    # Create environment
    env = TrafficEnv(
        port=port,
        model_name=model_name,
        model_idx=env_idx,
        sim_length=sim_length,
        base_gen_car_distrib=["uniform", 3000],
        num_of_episodes=100,
        state_representation=state_repr,
        action_strategy=action_strat,
        reward_function=reward_func,
        vsl_enforcement="recommend",
        sar_config=sar_config
    )
    
    return Monitor(env)


def run_experiment(experiment_name, state_name, action_name, reward_name, 
                   training_steps=50000, n_envs=4):
    """Run a training experiment with specific SAR configuration"""
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"State: {state_name}, Action: {action_name}, Reward: {reward_name}")
    print(f"{'='*60}\n")
    
    # SAR configuration
    sar_config = {
        'max_flow': 10000.0,
        'max_occupancy': 100.0,
        'max_queue_length': 575.0 * 3 / 7
    }
    
    # Create vectorized environment
    base_port = 8000
    envs = SubprocVecEnv([
        lambda i=i: create_env_with_sar(
            i, base_port + i, experiment_name, 3600,
            state_name, action_name, reward_name, sar_config
        )
        for i in range(n_envs)
    ])
    
    # Create and train model
    model = DQN(
        "MlpPolicy",
        envs,
        learning_rate=1e-4,
        buffer_size=100000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        verbose=1,
        device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    )
    
    # Train
    model.learn(total_timesteps=training_steps, progress_bar=True)
    
    # Save model
    model_path = f"./models/{experiment_name}.zip"
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Cleanup
    envs.close()
    
    return model


def evaluate_model(model, experiment_name, state_name, action_name, reward_name):
    """Evaluate a trained model"""
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {experiment_name}")
    print(f"{'='*60}\n")
    
    # Create evaluation environment
    sar_config = {
        'max_flow': 10000.0,
        'max_occupancy': 100.0,
        'max_queue_length': 575.0 * 3 / 7
    }
    
    eval_env = create_env_with_sar(
        0, 9000, f"{experiment_name}_eval", 7200,
        state_name, action_name, reward_name, sar_config
    )
    
    # Run evaluation
    obs, _ = eval_env.reset()
    total_reward = 0
    done = False
    steps = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = eval_env.step(action)
        total_reward += reward
        steps += 1
    
    # Get summary statistics
    summary = eval_env.logger.get_summary_statistics()
    
    print(f"Evaluation Results:")
    print(f"  Total Steps: {steps}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Avg Flow: {summary.get('avg_flow_vph', 0):.1f} veh/h")
    print(f"  Avg Speed: {summary.get('avg_speed_kph', 0):.1f} km/h")
    print(f"  Queue Length: {summary.get('avg_queue_m', 0):.1f} m")
    print(f"  Control Actions: {summary.get('total_control_actions', 0)}")
    
    eval_env.close()
    
    return summary


# ============================================================================
# MAIN EXPERIMENT SCRIPT
# ============================================================================

def main():
    """Run multiple experiments with different SAR configurations"""
    
    # Define experiments to run
    experiments = [
        # Standard configurations
        ("standard_balanced", "full_metrics", "absolute_speed", "balanced"),
        ("standard_mobility", "full_metrics", "absolute_speed", "mobility"),
        ("standard_safety", "full_metrics", "absolute_speed", "safety"),
        
        # Different state representations
        ("minimal_balanced", "minimal", "absolute_speed", "balanced"),
        
        # Different action strategies
        ("relative_balanced", "full_metrics", "relative_speed", "balanced"),
        
        # You can add custom components here once registered
        # ("compact_conservative", "compact", "conservative", "balanced"),
        # ("emission_focused", "full_metrics", "absolute_speed", "emission"),
    ]
    
    # Training parameters
    TRAINING_STEPS = 10000  # Reduced for example
    N_ENVS = 2  # Reduced for example
    
    # Run experiments
    results = {}
    
    for exp_name, state, action, reward in experiments:
        try:
            # Train model
            model = run_experiment(
                exp_name, state, action, reward,
                TRAINING_STEPS, N_ENVS
            )
            
            # Evaluate model
            summary = evaluate_model(model, exp_name, state, action, reward)
            results[exp_name] = summary
            
        except Exception as e:
            print(f"Error in experiment {exp_name}: {e}")
            results[exp_name] = {"error": str(e)}
    
    # Print comparison
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*60}\n")
    
    print(f"{'Experiment':<20} {'Avg Flow':<12} {'Avg Speed':<12} {'Avg Queue':<12} {'Total Reward':<12}")
    print("-" * 68)
    
    for exp_name, summary in results.items():
        if "error" in summary:
            print(f"{exp_name:<20} ERROR: {summary['error']}")
        else:
            flow = summary.get('avg_flow_vph', 0)
            speed = summary.get('avg_speed_kph', 0)
            queue = summary.get('avg_queue_m', 0)
            # Note: Total reward would need to be tracked separately
            print(f"{exp_name:<20} {flow:<12.1f} {speed:<12.1f} {queue:<12.1f}")


# ============================================================================
# REGISTERING CUSTOM COMPONENTS
# ============================================================================

def register_custom_components():
    """Register custom SAR components for use with factory functions"""
    
    # This would need to be added to sar_framework.py's factory functions
    # For now, we'll just demonstrate the pattern
    
    # Example of how you might extend the factory functions:
    custom_states = {
        'compact': CompactState,
    }
    
    custom_actions = {
        'conservative': ConservativeAction,
    }
    
    custom_rewards = {
        'emission': EmissionReward,
    }
    
    return custom_states, custom_actions, custom_rewards


if __name__ == "__main__":
    # Check SUMO
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        print("Please set SUMO_HOME environment variable")
        sys.exit(1)
    
    # Run experiments
    main()