# DRL-VSL Experiments Module

## Overview

The experiments module provides a comprehensive framework for systematically evaluating different State-Action-Reward (SAR) configurations in the DRL-VSL (Deep Reinforcement Learning for Variable Speed Limit) system. It enables researchers and practitioners to:

- Run controlled experiments with different SAR combinations
- Compare performance across various traffic scenarios
- Identify optimal configurations for specific objectives
- Generate detailed reports and visualizations
- Conduct ablation studies and hyperparameter tuning

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Components](#components)
4. [Running Experiments](#running-experiments)
5. [Comparing Results](#comparing-results)
6. [Configuration Files](#configuration-files)
7. [Output Interpretation](#output-interpretation)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

```bash
# Ensure you have the DRL-VSL framework installed
cd /path/to/drl-vsl-project

# Install additional dependencies for experiments
pip install pandas matplotlib seaborn pyyaml

# Verify SUMO installation
echo $SUMO_HOME  # Should point to your SUMO installation
```

### Directory Structure

```
experiments/
├── __init__.py
├── run_experiments.py      # Main experiment runner
├── compare_sar.py         # Comparison and analysis tools
├── configs/               # Experiment configurations
│   ├── mobility_configs.yaml
│   ├── safety_configs.yaml
│   └── custom_configs.yaml
└── README.md             # This file
```

## Quick Start

### 1. Run a Single Experiment

```bash
# Basic experiment with default settings
python experiments/run_experiments.py --single \
    --name my_first_experiment \
    --state full_metrics \
    --action absolute_speed \
    --reward balanced \
    --timesteps 50000
```

### 2. Run Multiple Experiments from Config

```bash
# Run all mobility-focused experiments
python experiments/run_experiments.py \
    --config experiments/configs/mobility_configs.yaml \
    --parallel --n-workers 4
```

### 3. Compare Results

```bash
# Generate comparison report
python experiments/compare_sar.py --action report

# View rankings
python experiments/compare_sar.py --action rank
```

## Components

### run_experiments.py

The main experiment runner that handles:

- **Training**: Trains DRL models with specified SAR configurations
- **Evaluation**: Tests models across different traffic scenarios
- **Metrics Collection**: Gathers performance data
- **Result Storage**: Saves all data for later analysis

#### Key Classes

- `ExperimentConfig`: Defines experiment parameters
- `ExperimentRunner`: Orchestrates experiment execution

### compare_sar.py

Analysis tools for comparing experiment results:

- **Loading Results**: Reads experiment data from disk
- **Ranking**: Ranks configurations based on multiple criteria
- **Visualization**: Creates plots and heatmaps
- **Reporting**: Generates comprehensive comparison reports

#### Key Classes

- `SARComparator`: Main analysis class with all comparison methods

### Configuration Files

YAML files defining experiment parameters:

- `mobility_configs.yaml`: Traffic flow optimization
- `safety_configs.yaml`: Safety-focused configurations
- `custom_configs.yaml`: User-defined experiments

## Running Experiments

### Command Line Interface

#### Single Experiment

```bash
python experiments/run_experiments.py --single \
    --name experiment_name \
    --state [full_metrics|minimal|density_focused|...] \
    --action [absolute_speed|relative_speed|traffic_adaptive|...] \
    --reward [mobility|safety|balanced|emission|...] \
    --timesteps 100000 \
    --n-envs 4 \
    --n-eval 5
```

#### Batch Experiments

```bash
# Run all SAR combinations
python experiments/run_experiments.py --all-sar \
    --timesteps 200000 \
    --parallel

# Run from configuration file
python experiments/run_experiments.py \
    --config path/to/config.yaml \
    --parallel --n-workers 8
```

#### Advanced Options

```bash
# Skip training and use existing models
python experiments/run_experiments.py --single \
    --name eval_only \
    --state full_metrics \
    --action absolute_speed \
    --reward mobility \
    --timesteps 0  # 0 means evaluation only

# Custom evaluation scenarios
python experiments/run_experiments.py --single \
    --name custom_eval \
    --scenarios low medium high congested \
    --n-eval 10
```

### Python API

```python
from experiments import ExperimentConfig, run_single_experiment, run_batch_experiments

# Single experiment
config = ExperimentConfig(
    name="api_test",
    state="full_metrics",
    action="absolute_speed",
    reward="mobility",
    timesteps=100000
)
results = run_single_experiment(config)

# Batch experiments
configs = [
    ExperimentConfig(name=f"test_{i}", state=s, action=a, reward=r)
    for i, (s, a, r) in enumerate([
        ("full_metrics", "absolute_speed", "mobility"),
        ("minimal", "relative_speed", "safety"),
        ("full_metrics", "traffic_adaptive", "balanced")
    ])
]
results = run_batch_experiments(configs, parallel=True)
```

## Comparing Results

### Generate Reports

```bash
# Comprehensive comparison report
python experiments/compare_sar.py --action report \
    --output analysis_report.json

# Specific experiments only
python experiments/compare_sar.py --action report \
    --experiments mobility_full_absolute safety_full_gradual \
    --output targeted_report.json
```

### Rank Configurations

```bash
# Default ranking (avg_reward + robustness)
python experiments/compare_sar.py --action rank

# Custom criteria and weights
python experiments/compare_sar.py --action rank \
    --criteria avg_reward robustness_score flow_downstream_high_traffic \
    --weights 0.5 0.3 0.2 \
    --output rankings.csv
```

### Visualizations

```bash
# Performance comparison plots
python experiments/compare_sar.py --action plot \
    --plot-type comparison \
    --metrics avg_reward robustness_score \
    --output comparison_plots.png

# Heatmap of SAR combinations
python experiments/compare_sar.py --action plot \
    --plot-type heatmap \
    --metrics avg_reward \
    --output performance_heatmap.png

# Scenario performance
python experiments/compare_sar.py --action plot \
    --plot-type scenario \
    --output scenario_performance.png
```

### Export Data

```bash
# Export all metrics to CSV
python experiments/compare_sar.py --action export \
    --output all_metrics.csv
```

## Configuration Files

### Structure

Configuration files use YAML format with three possible structures:

#### 1. Single Experiment

```yaml
name: "single_experiment"
state: "full_metrics"
action: "absolute_speed"
reward: "balanced"
timesteps: 100000
eval_scenarios:
  - name: "rush_hour"
    demand: 4500
```

#### 2. Multiple Experiments

```yaml
experiments:
  - name: "exp1"
    state: "full_metrics"
    action: "absolute_speed"
    reward: "mobility"
    
  - name: "exp2"
    state: "minimal"
    action: "relative_speed"
    reward: "safety"
```

#### 3. Base Config with Variations

```yaml
base_config:
  algorithm: "DQN"
  timesteps: 200000
  n_train_envs: 4

variations:
  - name: "var1"
    state: "full_metrics"
    action: "absolute_speed"
    reward: "mobility"
    
  - name: "var2"
    state: "minimal"
    action: "relative_speed"
    reward: "safety"
```

### Available Options

#### States
- `full_metrics`: Complete 9-feature state
- `minimal`: Simplified 3-feature state
- `density_focused`: Traffic density metrics
- `multi_segment_density`: Spatial density awareness
- `queue_length`: Queue-focused state
- `queue_dynamics`: Advanced queue metrics

#### Actions
- `absolute_speed`: Direct speed limit setting
- `relative_speed`: Incremental changes
- `traffic_adaptive`: Adapts to traffic conditions
- `occupancy_based`: Occupancy-driven decisions
- `gradual_change`: Smooth transitions
- `momentum_based`: Momentum-aware changes

#### Rewards
- `mobility`: Maximize traffic flow
- `safety`: Minimize risk and variance
- `balanced`: Multi-objective optimization
- `emission`: Reduce emissions
- `fuel_efficiency`: Minimize fuel consumption
- `passenger_comfort`: Smooth ride quality
- `smooth_flow`: Traffic harmonization

## Output Interpretation

### Experiment Results Structure

```
experiment_results/
├── experiment_name/
│   ├── YYYYMMDD_HHMMSS/
│   │   ├── config.json          # Experiment configuration
│   │   ├── results.json         # Complete results
│   │   ├── model.zip           # Trained model
│   │   └── tensorboard/        # Training logs
```

### Results JSON Format

```json
{
  "config": {
    "name": "experiment_name",
    "state": "full_metrics",
    "action": "absolute_speed",
    "reward": "mobility"
  },
  "training": {
    "model_path": "path/to/model.zip",
    "training_time": 3600.5,
    "timesteps": 200000
  },
  "evaluation": {
    "low_traffic": {
      "demand": 2000,
      "rewards": {
        "mean": 0.85,
        "std": 0.12
      },
      "metrics": {
        "flow_downstream": {...},
        "avg_speed_before": {...},
        "queue_length_upstream": {...}
      }
    }
  },
  "metrics": {
    "avg_reward": 0.82,
    "robustness_score": 0.75,
    "reward_low_traffic": 0.85,
    "reward_high_traffic": 0.78
  }
}
```

### Key Metrics

- **avg_reward**: Average reward across all scenarios
- **robustness_score**: Inverse of performance variance (higher = more consistent)
- **flow_downstream**: Traffic throughput (vehicles/hour)
- **avg_speed_before**: Average speed in controlled zone
- **queue_length_upstream**: Queue length before merge
- **collisions**: Safety metric

## Best Practices

### 1. Experiment Design

- **Start Small**: Test with short training times first
- **Use Configs**: Define experiments in YAML for reproducibility
- **Systematic Testing**: Test one component at a time for clear insights

### 2. Resource Management

```bash
# Monitor resource usage
htop  # CPU and memory
nvidia-smi  # GPU usage (if applicable)

# Parallel execution guidelines
# n_workers = min(available_cores - 1, number_of_experiments)
python experiments/run_experiments.py --parallel --n-workers 4
```

### 3. Result Organization

```bash
# Create meaningful experiment names
--name "mobility_heavy_traffic_v2"

# Use consistent naming conventions
# Format: [objective]_[state]_[action]_[version]
```

### 4. Analysis Workflow

1. Run baseline experiments first
2. Compare variations against baseline
3. Identify best performers
4. Run extended training for top configurations
5. Validate with diverse scenarios

## Troubleshooting

### Common Issues

#### SUMO Connection Errors
```bash
# Check SUMO installation
echo $SUMO_HOME

# Ensure unique ports for parallel execution
# Experiments automatically assign ports starting from 8000
```

#### Memory Issues
```bash
# Reduce parallel workers
--n-workers 2

# Reduce number of training environments
--n-envs 2

# Use smaller buffer size in hyperparameters
```

#### Missing Results
```bash
# Check experiment directory
ls experiment_results/

# Look for error logs
grep -r "error" experiment_results/*/YYYYMMDD_*/results.json
```

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run single experiment with minimal settings
python experiments/run_experiments.py --single \
    --name debug_test \
    --timesteps 1000 \
    --n-envs 1 \
    --n-eval 1
```

## Advanced Usage

### Custom SAR Components

```python
# Use custom SAR components from sar_components module
config = ExperimentConfig(
    name="custom_sar",
    state="multi_segment_density",  # From sar_components
    action="traffic_adaptive",      # From sar_components  
    reward="smooth_flow",          # From sar_components
    timesteps=200000
)
```

### Hyperparameter Tuning

```yaml
# In config file
hyperparams:
  learning_rate: 0.0005
  buffer_size: 100000
  batch_size: 64
  exploration_fraction: 0.25
  exploration_final_eps: 0.05
```

### Multi-Stage Experiments

```bash
# Stage 1: Quick exploration
python experiments/run_experiments.py --all-sar --timesteps 50000

# Stage 2: Refined training for top performers
python experiments/compare_sar.py --action rank --output top_configs.csv
# Manually select top 5 configurations

# Stage 3: Extended training
python experiments/run_experiments.py --config extended_training.yaml
```

## Contributing

When adding new experiments or features:

1. Document new SAR components in configuration files
2. Update this README with new options
3. Add example configurations
4. Include expected performance baselines

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review example configurations
3. Examine the source code documentation
4. Create an issue with:
   - Error messages
   - Configuration used
   - System specifications