# SUMO Configuration System

## Overview

The SUMO configuration system provides a flexible, YAML-based approach to managing SUMO simulation parameters in the DRL-VSL framework. This allows you to:

- Easily switch between different SUMO configurations
- Create experiment-specific settings
- Use presets for common scenarios (training, evaluation, debug, performance)
- Override parameters without modifying code
- Inherit configurations for easy customization

## Directory Structure

```
traffic_environment/sumo/sumo_configs/
├── default_sumo_config.yaml       # Base configuration
├── training_sumo_config.yaml      # Optimized for training
├── evaluation_sumo_config.yaml    # For evaluation with GUI
├── debug_sumo_config.yaml         # Detailed logging and output
├── performance_sumo_config.yaml   # Maximum speed
└── experiment_specific/
    └── high_fidelity.yaml        # Maximum realism

core/
└── sumo_config.py                # Configuration loader module
```

## Quick Start

### 1. Setup

Run the setup script from your project root:

```bash
python setup_sumo_config.py
```

This creates:
- Configuration directories
- Default YAML files
- Updates necessary imports

### 2. Using Presets

```python
# In experiments
python experiments/run_experiments.py --single \
    --name test \
    --sumo-preset training  # Use training preset

# In code
env = TrafficEnv(
    ...,
    sumo_preset="evaluation"  # Use evaluation preset
)
```

Available presets:
- `training`: No GUI, faster simulation, deterministic
- `evaluation`: GUI enabled, detailed metrics
- `debug`: Maximum logging, state saves
- `performance`: Optimized for speed

### 3. Using Config Files

```python
# In experiments
python experiments/run_experiments.py --single \
    --name test \
    --sumo-config traffic_environment/sumo/sumo_configs/debug_sumo_config.yaml

# In code
env = TrafficEnv(
    ...,
    sumo_config="traffic_environment/sumo/sumo_configs/evaluation_sumo_config.yaml"
)
```

### 4. In Experiment Configs

```yaml
# experiments/configs/custom_configs.yaml
experiments:
  - name: "high_fidelity_safety"
    state: "full_metrics"
    action: "absolute_speed"
    reward: "safety"
    sumo_config: "traffic_environment/sumo/sumo_configs/experiment_specific/high_fidelity.yaml"
    
  - name: "fast_mobility_training"
    state: "minimal"
    action: "absolute_speed"
    reward: "mobility"
    sumo_preset: "performance"
```

## Configuration Structure

### Basic Structure

```yaml
sumo:
  binary: "sumo-gui"              # SUMO executable (sumo or sumo-gui)
  step_length: 1.0                # Simulation step in seconds
  default_action_step_length: 0.2 # Vehicle action frequency
  start: true                     # Start simulation immediately
  
  # Safety parameters
  default_emergencydecel: 7.0     # Emergency braking (m/s²)
  time_to_teleport: -1           # -1 disables teleporting
  collision_action: "warn"        # How to handle collisions
  
  # Traffic parameters
  random_depart_offset: 3600      # Randomize departure times
  lateral_resolution: 0.2         # Lane changing precision
  
  # Logging
  no_step_log: true              # Disable step logging
  no_warnings: true              # Suppress warnings
  verbose: false                 # Verbose output
  
  # Additional SUMO options
  additional_options:
    - "--seed=42"
    - "--device.emissions.probability=1.0"

files:
  net_file: "../3_2_merge.net.xml"
  additional_files: "../loops_detectors.add.xml"
  gui_settings_file: "../colored.view.xml"

paths:
  sumo_home_env: "SUMO_HOME"
  generated_configs_dir: "traffic_environment/sumo/generated_configs"
  generated_flows_dir: "traffic_environment/sumo/generated_flows"
  sumo_logs_dir: "logs/sumo_log"
```

### Inheritance

Use `extends` to inherit from another config:

```yaml
extends: "default_sumo_config.yaml"

# Override specific values
sumo:
  binary: "sumo"  # No GUI
  step_length: 2.0  # Larger steps for speed
```

## Advanced Usage

### Custom Configuration

Create your own config file:

```yaml
# traffic_environment/sumo/sumo_configs/my_custom_config.yaml
extends: "default_sumo_config.yaml"

sumo:
  binary: "sumo"
  step_length: 0.5
  
  # Custom safety settings
  default_emergencydecel: 9.0
  time_to_teleport: 180
  
  # Enable specific outputs
  additional_options:
    - "--tripinfo-output=tripinfo.xml"
    - "--emission-output=emissions.xml"
    - "--queue-output=queue.xml"
    - "--statistic-output=stats.xml"
```

### Programmatic Configuration

```python
from core.sumo_config import SumoConfig, load_sumo_config

# Load from file
config = load_sumo_config("traffic_environment/sumo/sumo_configs/training_sumo_config.yaml")

# Modify at runtime
config.update_config({
    'sumo': {
        'step_length': 0.5,
        'verbose': True
    }
})

# Use in environment
env = TrafficEnv(..., sumo_config=config)

# Or update existing environment
env.update_sumo_config({
    'sumo': {'time_to_teleport': 300}
})
```

### Dynamic Preset Switching

```python
# Start with training preset
env = TrafficEnv(..., sumo_preset="training")

# Switch to debug mode
env.set_sumo_preset("debug")

# Continue simulation with new settings
obs, info = env.reset()
```

## Parameter Reference

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `binary` | `sumo-gui` | SUMO executable (`sumo` or `sumo-gui`) |
| `step_length` | `1.0` | Simulation time step in seconds |
| `default_action_step_length` | `0.2` | How often vehicles update their behavior |
| `time_to_teleport` | `-1` | Time before stuck vehicles teleport (-1 disables) |
| `collision_action` | `warn` | How to handle collisions: `none`, `warn`, `teleport`, `remove` |
| `default_emergencydecel` | `7.0` | Maximum emergency deceleration (m/s²) |
| `random_depart_offset` | `3600` | Randomize vehicle departures (seconds) |
| `lateral_resolution` | `0.2` | Precision of lateral movement |

### Performance Impact

- **`step_length`**: Smaller = more accurate but slower
- **`time_to_teleport`**: -1 = realistic but can cause deadlocks
- **`collision_action`**: `none` = fastest but unrealistic
- **`lateral_resolution`**: Higher values = faster but less precise lane changes
- **`binary`**: `sumo` = faster (no GUI), `sumo-gui` = visual but slower

## Best Practices

### 1. Training vs Evaluation

**Training**: Focus on speed and consistency
```yaml
sumo:
  binary: "sumo"  # No GUI
  time_to_teleport: 300  # Prevent deadlocks
  additional_options:
    - "--seed=42"  # Reproducibility
```

**Evaluation**: Focus on realism and metrics
```yaml
sumo:
  binary: "sumo-gui"  # Visualization
  time_to_teleport: -1  # Realistic behavior
  verbose: true  # Detailed output
```

### 2. Debugging Issues

Use debug preset when encountering problems:
```bash
python experiments/run_experiments.py --single \
    --name debug_test \
    --sumo-preset debug \
    --timesteps 1000
```

Check generated files:
- `collisions.xml`: Collision details
- `stats.xml`: Simulation statistics
- `tripinfo.xml`: Vehicle trip information

### 3. Performance Optimization

For large-scale experiments:
```yaml
sumo:
  binary: "sumo"
  step_length: 2.0  # Larger steps
  collision_action: "none"  # Skip collision checks
  no_step_log: true
  no_warnings: true
  additional_options:
    - "--no-internal-links"
    - "--max-depart-delay=0"
```

### 4. Experiment-Specific Configs

Create specialized configs for different research questions:

```yaml
# For emission studies
extends: "evaluation_sumo_config.yaml"
sumo:
  additional_options:
    - "--device.emissions.probability=1.0"
    - "--emission-output=emissions.xml"
    - "--device.emissions.vehicleMass=1500"

# For safety studies
extends: "evaluation_sumo_config.yaml"
sumo:
  default_emergencydecel: 9.0
  collision_action: "warn"
  additional_options:
    - "--collision.output=collisions.xml"
    - "--collision.check-junctions"
    - "--collision.mingap-factor=1.0"
```

## Troubleshooting

### SUMO Binary Not Found
```python
# Specify explicitly
env = TrafficEnv(
    ...,
    sumo_binary_path_override="/usr/local/bin/sumo-gui"
)
```

### Configuration Not Loading
Check file paths:
```python
import os
print(os.path.exists("traffic_environment/sumo/sumo_configs/default_sumo_config.yaml"))
```

### Parameters Not Applied
Enable verbose logging:
```yaml
sumo:
  verbose: true
  no_warnings: false
```

## Integration with Experiments

The SUMO configuration system is fully integrated with the experiments module:

```bash
# Command line
python experiments/run_experiments.py \
    --single \
    --name test \
    --sumo-config path/to/config.yaml

# Or use preset
python experiments/run_experiments.py \
    --single \
    --name test \
    --sumo-preset training
```

In batch experiments:
```yaml
base_config:
  sumo_preset: "training"  # Default for all

variations:
  - name: "detailed_analysis"
    sumo_config: "traffic_environment/sumo/sumo_configs/evaluation_sumo_config.yaml"
```

## Future Extensions

The configuration system can be extended with:

1. **Scenario-specific configs**: Traffic jam, weather conditions
2. **Network-specific settings**: Different road networks
3. **Vehicle mix configurations**: CAV percentages, truck ratios
4. **Output specifications**: Which data to collect
5. **Calibration profiles**: Real-world traffic patterns

The modular design makes it easy to add new parameters and presets as needed.