# setup_sumo_config.py
"""
Setup script to create SUMO configuration structure and files.
Run this from your project root directory.
"""

import os
import yaml
from pathlib import Path

def create_sumo_config_structure():
    """Create the SUMO configuration directory structure and files."""
    
    output_dir = os.path.abspath(os.path.join("traffic_environment", "sumo", "sumo_configs"))
    os.makedirs(output_dir, exist_ok=True)

    # Create directories
    config_dirs = [
        f"{output_dir}",
        f"{output_dir}/experiment_specific",
        f"{output_dir}/experiments"
    ]
    
    print("Creating SUMO configuration directories...")
    for dir_path in config_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {dir_path}")
    
    # Default SUMO configuration
    default_config = {
        'sumo': {
            'binary': 'sumo-gui',
            'step_length': 1.0,
            'default_action_step_length': 0.2,
            'start': True,
            'default_emergencydecel': 7.0,
            'time_to_teleport': -1,
            'collision_action': 'warn',
            'random_depart_offset': 3600,
            'lateral_resolution': 0.2,
            'no_step_log': True,
            'no_warnings': True,
            'verbose': False,
            'additional_options': []
        },
        'files': {
            'net_file': '../3_2_merge.net.xml',
            'additional_files': '../loops_detectors.add.xml',
            'gui_settings_file': '../colored.view.xml'
        },
        'paths': {
            'sumo_home_env': 'SUMO_HOME',
            'generated_configs_dir': 'traffic_environment/sumo/generated_configs',
            'generated_flows_dir': 'traffic_environment/sumo/generated_flows',
            'sumo_logs_dir': 'logs/sumo_log'
        }
    }
    
    # Training configuration
    training_config = {
        'extends': 'default_sumo_config.yaml',
        'sumo': {
            'binary': 'sumo',
            'step_length': 1.0,
            'default_action_step_length': 0.5,
            'time_to_teleport': 300,
            'no_step_log': True,
            'no_warnings': True,
            'verbose': False,
            'random_depart_offset': 1800,
            'additional_options': [
                '--seed=42',
                '--no-internal-links'
            ]
        }
    }
    
    # Evaluation configuration
    evaluation_config = {
        'extends': 'default_sumo_config.yaml',
        'sumo': {
            'binary': 'sumo-gui',
            'step_length': 0.5,
            'default_action_step_length': 0.1,
            'time_to_teleport': -1,
            'collision_action': 'warn',
            'no_step_log': False,
            'no_warnings': False,
            'verbose': True,
            'random_depart_offset': 7200,
            'additional_options': [
                '--collision.check-junctions',
                '--collision.mingap-factor=1.0',
                '--device.emissions.probability=1.0'
            ]
        }
    }
    
    # Debug configuration
    debug_config = {
        'extends': 'default_sumo_config.yaml',
        'sumo': {
            'binary': 'sumo-gui',
            'step_length': 1.0,
            'collision_action': 'warn',
            'no_step_log': False,
            'no_warnings': False,
            'verbose': True,
            'time_to_teleport': 60,
            'additional_options': [
                '--collision.output=collisions.xml',
                '--statistic-output=stats.xml',
                '--tripinfo-output=tripinfo.xml',
                '--lanedata-output=lanedata.xml',
                '--save-state.times=1800,3600'
            ]
        }
    }
    
    # Performance configuration
    performance_config = {
        'extends': 'default_sumo_config.yaml',
        'sumo': {
            'binary': 'sumo',
            'step_length': 2.0,
            'default_action_step_length': 1.0,
            'time_to_teleport': 120,
            'collision_action': 'none',
            'no_step_log': True,
            'no_warnings': True,
            'verbose': False,
            'lateral_resolution': 1.0,
            'additional_options': [
                '--no-internal-links',
                '--ignore-junction-blocker=10',
                '--max-depart-delay=0',
                '--device.rerouting.probability=0'
            ]
        }
    }
    
    # High fidelity configuration
    high_fidelity_config = {
        'extends': '../default_sumo_config.yaml',
        'sumo': {
            'binary': 'sumo-gui',
            'step_length': 0.1,
            'default_action_step_length': 0.1,
            'default_emergencydecel': 9.0,
            'time_to_teleport': -1,
            'collision_action': 'warn',
            'lateral_resolution': 0.1,
            'no_step_log': False,
            'no_warnings': False,
            'verbose': True,
            'additional_options': [
                '--device.emissions.probability=1.0',
                '--device.fcd.probability=1.0',
                '--fcd-output=fcd.xml',
                '--emission-output=emissions.xml',
                '--full-output=full.xml',
                '--collision.check-junctions',
                '--collision.mingap-factor=1.0',
                '--pedestrian.model=striping',
                '--lateral-resolution-sublanes=0.5'
            ]
        }
    }
    
    # Write configuration files

    configs_to_write = [
        (f'{output_dir}/default_sumo_config.yaml', default_config),
        (f'{output_dir}/training_sumo_config.yaml', training_config),
        (f'{output_dir}/evaluation_sumo_config.yaml', evaluation_config),
        (f'{output_dir}/debug_sumo_config.yaml', debug_config),
        (f'{output_dir}/performance_sumo_config.yaml', performance_config),
        (f'{output_dir}/experiment_specific/high_fidelity.yaml', high_fidelity_config),
    ]
    
    print("\nCreating SUMO configuration files...")
    for file_path, config_data in configs_to_write:
        with open(file_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        print(f"  ✓ Created: {file_path}")
    
    # Create the sumo_config.py module if it doesn't exist
    sumo_config_path = Path("core/sumo_config.py")
    if not sumo_config_path.exists():
        print(f"\n⚠️  Note: core/sumo_config.py doesn't exist yet.")
        print("  Copy the sumo_config.py content from the artifacts above.")
    
    # Update core/__init__.py to include sumo_config
    core_init_path = Path("core/__init__.py")
    if core_init_path.exists():
        with open(core_init_path, 'r') as f:
            content = f.read()
        
        if 'sumo_config' not in content:
            # Add import
            import_lines = [
                "",
                "# SUMO configuration",
                "try:",
                "    from .sumo_config import (",
                "        SumoConfig,",
                "        load_sumo_config,",
                "        get_default_sumo_config,",
                "        get_preset_config,",
                "        PRESETS",
                "    )",
                "except ImportError:",
                "    # Sumo config module not available yet",
                "    SumoConfig = None",
                "    load_sumo_config = None",
                "    get_default_sumo_config = None",
                "    get_preset_config = None",
                "    PRESETS = None",
                ""
            ]
            
            # Add to __all__
            if "__all__" in content:
                # Insert before __all__
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith("__all__"):
                        lines[i:i] = import_lines
                        break
                
                # Add to __all__ list
                for i, line in enumerate(lines):
                    if "__all__" in line and "]" in line:
                        lines[i] = line.replace("]", ",\n    \n    # SUMO Configuration\n    \"SumoConfig\",\n    \"load_sumo_config\",\n    \"get_default_sumo_config\",\n    \"get_preset_config\",\n    \"PRESETS\",\n]")
                        break
                
                content = '\n'.join(lines)
            else:
                content += '\n'.join(import_lines)
            
            with open(core_init_path, 'w') as f:
                f.write(content)
            print("\n✓ Updated core/__init__.py with SUMO config imports")
    
    print("\n✅ SUMO configuration setup complete!")
    print("\nExample usage in experiments:")
    print(f"""
# In experiment config YAML:
experiments:
  - name: "fast_training"
    state: "full_metrics"
    action: "absolute_speed"
    reward: "mobility"
    sumo_preset: "performance"  # Use performance preset
    
  - name: "detailed_eval"
    state: "full_metrics"
    action: "absolute_speed"
    reward: "safety"
    sumo_config: "{output_dir}/experiment_specific/high_fidelity.yaml"

# Or programmatically:
from core.sumo_config import get_preset_config

env = TrafficEnv(
    ...,
    sumo_preset="training"  # Use training preset
)

# Or with custom config:
env = TrafficEnv(
    ...,
    sumo_config="path/to/custom_config.yaml"
)
""")


if __name__ == "__main__":
    create_sumo_config_structure()