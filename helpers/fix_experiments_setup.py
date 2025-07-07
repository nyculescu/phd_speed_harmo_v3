# fix_experiments_setup.py
"""
Script to fix the experiments setup and run a test experiment.
Place this in your project root directory.
"""

import os
import sys
from pathlib import Path
import shutil

def main():
    print("Fixing DRL-VSL Experiments Setup")
    print("=" * 50)
    
    # 1. Fix the import in drl_vsl_integration.py
    integration_file = Path("core/drl_vsl_integration.py")
    if integration_file.exists():
        print(f"Fixing imports in {integration_file}...")
        
        # Read the file
        with open(integration_file, 'r') as f:
            content = f.read()
        
        # Check if the fix is needed
        if "from .drl_vsl_refactored import TrafficEnv" in content and ", create_sumocfg" not in content:
            # Apply the fix
            content = content.replace(
                "from .drl_vsl_refactored import TrafficEnv",
                "from .drl_vsl_refactored import TrafficEnv, create_sumocfg"
            )
            
            # Write back
            with open(integration_file, 'w') as f:
                f.write(content)
            
            print("✓ Fixed import in drl_vsl_integration.py")
        else:
            print("✓ Import already fixed or different format")
    else:
        print("✗ Could not find core/drl_vsl_integration.py")
        return
    
    # 2. Create the experiments/experiment_results directory
    exp_results_dir = Path("experiments/experiment_results")
    exp_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {exp_results_dir}")
    
    # 3. Create other necessary directories
    directories = [
        "logs",
        "rl_models",
        "rl_models/optuna_params",
        "traffic_environment/sumo/generated_flows",
        "traffic_environment/sumo/generated_configs",
        "logs/sumo_log"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created/verified: {dir_path}")
    
    # 4. Check SUMO
    if 'SUMO_HOME' not in os.environ:
        print("\n⚠️  WARNING: SUMO_HOME environment variable not set!")
        print("Please set SUMO_HOME to your SUMO installation directory")
        print("Example: export SUMO_HOME=/usr/share/sumo")
        print("\nContinuing without SUMO check...")
    else:
        print(f"\n✓ SUMO_HOME found: {os.environ['SUMO_HOME']}")
    
    # 5. Test import
    print("\nTesting imports...")
    try:
        sys.path.insert(0, str(Path.cwd()))
        from experiments.run_experiments import ExperimentConfig, run_single_experiment
        from experiments.compare_sar import SARComparator
        print("✓ Experiments module imports successful")
        
        # Test the fixed import
        from core.drl_vsl_integration import TrafficEnvCompat
        print("✓ Integration module imports successful")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nMake sure you have installed all dependencies:")
        print("pip install stable-baselines3 pandas matplotlib seaborn pyyaml torch")
        return
    
    print("\n" + "=" * 50)
    print("Setup complete! You can now run experiments:")
    print("\n1. Quick test (5 minutes):")
    print("   python experiments/run_experiments.py --single --name quick_test --timesteps 10000")
    print("\n2. Full experiment (30-60 minutes):")
    print("   python experiments/run_experiments.py --single --name full_test --timesteps 100000")
    print("\n3. Compare results (after running experiments):")
    print("   python experiments/compare_sar.py --action report")
    print("\n4. Run from config file:")
    print("   python experiments/run_experiments.py --config experiments/configs/mobility_configs.yaml")


if __name__ == "__main__":
    main()