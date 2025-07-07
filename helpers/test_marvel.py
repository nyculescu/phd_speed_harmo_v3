#!/usr/bin/env python3
"""
Quick test script to verify MARVEL integration
Run this before starting training to ensure everything is properly connected.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.sar_framework import (
    create_state_representation, 
    create_action_strategy,
    create_reward_function,
    TrafficMetrics
)

def test_marvel_components():
    """Test MARVEL component creation and basic functionality."""
    
    print("=" * 60)
    print("MARVEL Integration Test")
    print("=" * 60)
    
    # Test configuration
    test_config = {
        'max_flow': 10000.0,
        'max_occupancy': 100.0,
        'max_queue_length': 246.4
    }
    
    try:
        # Test State
        print("\n1. Testing MARVEL State...")
        marvel_state = create_state_representation('marvel', test_config)
        print(f"   ✓ Created MARVEL state")
        print(f"   - Features: {marvel_state.num_features}")
        print(f"   - Observation space: {marvel_state.get_observation_space()}")
        
        # Test state building
        test_metrics = TrafficMetrics()
        test_metrics.avg_speed_before = 25.0  # m/s
        test_metrics.occupancy_upstream = 30.0
        
        marvel_state.set_downstream_action(96.6)  # 60 mph in km/h
        raw_state = marvel_state.build_state(test_metrics)
        norm_state = marvel_state.preprocess_state(raw_state)
        print(f"   - Raw state: {raw_state}")
        print(f"   - Normalized state: {norm_state}")
        
        # Test Action
        print("\n2. Testing MARVEL Action...")
        marvel_action = create_action_strategy('marvel_speed', test_config)
        print(f"   ✓ Created MARVEL action strategy")
        print(f"   - Action space: {marvel_action.get_action_space()}")
        print(f"   - Speed mappings:")
        for action, speed in marvel_action.speed_actions.items():
            print(f"     Action {action}: {speed:.1f} km/h ({speed/1.609:.0f} mph)")
        
        # Test action application
        new_speed, penalty = marvel_action.apply_action(2, 96.6)  # Action 2 = 50 mph
        print(f"   - Test: action=2, current=96.6 km/h → new={new_speed:.1f} km/h, penalty={penalty}")
        
        # Test Reward
        print("\n3. Testing MARVEL Reward...")
        marvel_reward = create_reward_function('marvel', test_config)
        print(f"   ✓ Created MARVEL reward function")
        print(f"   - Weights: adaptability={marvel_reward.w1}, safety={marvel_reward.w2}, mobility={marvel_reward.w3}")
        
        # Test reward calculation
        test_metrics.current_speed_limit = 80.5  # 50 mph
        marvel_reward.set_downstream_action(64.4)  # 40 mph
        reward = marvel_reward.calculate(test_metrics)
        print(f"   - Test reward: {reward:.3f}")
        
        # Test all factory functions
        print("\n4. Testing All Factory Functions...")
        all_states = ['full_metrics', 'minimal', 'marvel']
        all_actions = ['absolute_speed', 'relative_speed', 'marvel_speed']
        all_rewards = ['mobility', 'safety', 'balanced', 'marvel']
        
        errors = []
        for state in all_states:
            try:
                s = create_state_representation(state, test_config)
                print(f"   ✓ State '{state}' OK")
            except Exception as e:
                errors.append(f"State '{state}': {e}")
                print(f"   ✗ State '{state}' FAILED: {e}")
        
        for action in all_actions:
            try:
                a = create_action_strategy(action, test_config)
                print(f"   ✓ Action '{action}' OK")
            except Exception as e:
                errors.append(f"Action '{action}': {e}")
                print(f"   ✗ Action '{action}' FAILED: {e}")
        
        for reward in all_rewards:
            try:
                r = create_reward_function(reward, test_config)
                print(f"   ✓ Reward '{reward}' OK")
            except Exception as e:
                errors.append(f"Reward '{reward}': {e}")
                print(f"   ✗ Reward '{reward}' FAILED: {e}")
        
        print("\n" + "=" * 60)
        if errors:
            print("MARVEL Integration Test: FAILED")
            print(f"Found {len(errors)} errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print("MARVEL Integration Test: PASSED")
            print("All components are properly integrated!")
            return True
            
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_yaml_config():
    """Test loading and validating YAML configuration."""
    print("\n" + "=" * 60)
    print("Testing YAML Configuration")
    print("=" * 60)
    
    try:
        # Import here to avoid issues if not all modules are set up
        from training.drl_vsl_train import Config
        
        config_path = "training/config/drl_vsl_train_config.yaml"
        
        if not Path(config_path).exists():
            print(f"✗ Config file not found: {config_path}")
            print("  Make sure you're running from the project root directory")
            return False
            
        config = Config(config_path)
        print(f"✓ Successfully loaded configuration from {config_path}")
        
        # Check for MARVEL in valid options
        if 'custom_combinations' in config.data['sar_config']:
            print(f"✓ Found {len(config.data['sar_config']['custom_combinations'])} custom combinations:")
            for combo in config.data['sar_config']['custom_combinations']:
                print(f"  - {combo['name']}: state={combo['state']}, action={combo['action']}, reward={combo['reward']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load/validate config: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nRunning MARVEL integration tests...\n")
    
    # Test components
    components_ok = test_marvel_components()
    
    # Test YAML config
    config_ok = test_yaml_config()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Components: {'PASSED' if components_ok else 'FAILED'}")
    print(f"  Configuration: {'PASSED' if config_ok else 'FAILED'}")
    print("=" * 60)
    
    if components_ok and config_ok:
        print("\n✓ All tests passed! You can now run training with:")
        print("  python training/drl_vsl_train.py --config training/config/drl_vsl_train_config.yaml")
    else:
        print("\n✗ Some tests failed. Please fix the issues before training.")
        sys.exit(1)