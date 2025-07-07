#!/usr/bin/env python3
"""
run_experiments.py - Helper script to run specific DRL-VSL experiments
"""

import subprocess
import sys
import time
from pathlib import Path
import argparse
import yaml

def load_config(config_path):
    """Load YAML configuration to see available combinations."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def list_combinations(config_path):
    """List all available custom combinations."""
    config = load_config(config_path)
    combos = config.get('sar_config', {}).get('custom_combinations', [])
    
    print("\nAvailable combinations:")
    for i, combo in enumerate(combos):
        print(f"  [{i}] {combo['name']:<20} - state: {combo['state']:<15} "
              f"action: {combo['action']:<15} reward: {combo['reward']}")
    
    return combos

def run_experiment(combo_name, config_path, dry_run=False):
    """Run a single experiment by name."""
    cmd = [
        sys.executable,
        "training/drl_vsl_train.py",
        "--config", str(config_path),
        "--combo-name", combo_name
    ]
    
    print(f"\nRunning experiment: {combo_name}")
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("(Dry run - not executing)")
        return
    
    start_time = time.time()
    
    try:
        process = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ Experiment '{combo_name}' completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment '{combo_name}' failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Experiment '{combo_name}' interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run specific DRL-VSL experiments")
    parser.add_argument('--config', type=str, 
                       default='training/config/drl_vsl_train_config.yaml',
                       help='Path to YAML configuration file')
    parser.add_argument('--list', action='store_true',
                       help='List available combinations and exit')
    parser.add_argument('--run', type=str, nargs='+',
                       help='Names of experiments to run')
    parser.add_argument('--run-index', type=int, nargs='+',
                       help='Indices of experiments to run')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')
    parser.add_argument('--sequential', action='store_true',
                       help='Run experiments sequentially with status summary')
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # List combinations
    combos = list_combinations(config_path)
    
    if args.list:
        sys.exit(0)
    
    # Determine which experiments to run
    experiments_to_run = []
    
    if args.run:
        experiments_to_run.extend(args.run)
    
    if args.run_index:
        for idx in args.run_index:
            if 0 <= idx < len(combos):
                experiments_to_run.append(combos[idx]['name'])
            else:
                print(f"Warning: Index {idx} out of range, skipping")
    
    if not experiments_to_run:
        print("\nNo experiments specified. Use --run or --run-index to select experiments.")
        print("Example: python run_experiments.py --run marvel_pure baseline_balanced")
        sys.exit(1)
    
    # Run experiments
    print(f"\nWill run {len(experiments_to_run)} experiments:")
    for exp in experiments_to_run:
        print(f"  - {exp}")
    
    if args.dry_run:
        print("\n(Dry run mode - no experiments will be executed)")
        return
    
    # Confirm before running
    response = input("\nProceed? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Run experiments
    results = {}
    for exp_name in experiments_to_run:
        success = run_experiment(exp_name, config_path, args.dry_run)
        results[exp_name] = success
        
        if args.sequential and not success:
            response = input("\nContinue with remaining experiments? [y/N]: ")
            if response.lower() != 'y':
                break
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for exp_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{exp_name:<30} {status}")
    
    successful = sum(1 for s in results.values() if s)
    print(f"\nCompleted: {successful}/{len(results)} experiments successful")

if __name__ == "__main__":
    main()