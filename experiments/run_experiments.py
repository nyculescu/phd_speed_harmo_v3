# experiments/run_experiments.py
"""
Main experiment runner for systematic evaluation of SAR configurations.
"""

import os
import sys
import json
import yaml
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from core.drl_vsl_integration import create_train_env_compat, create_eval_env_compat
from core.sar_framework import create_state_representation, create_action_strategy, create_reward_function
from training.drl_vsl_train import train_model, load_hyperparameters, load_sar_config

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    state: str
    action: str
    reward: str
    algorithm: str = "DQN"
    vsl_mode: str = "recommend"
    timesteps: int = 100_000
    n_train_envs: int = 2
    n_eval_episodes: int = 5
    eval_scenarios: List[Dict[str, Any]] = None
    hyperparams: Dict[str, Any] = None
    description: str = ""
    
    def __post_init__(self):
        if self.eval_scenarios is None:
            self.eval_scenarios = [
                {"demand": 2000, "name": "low_traffic"},
                {"demand": 3500, "name": "medium_traffic"},
                {"demand": 5000, "name": "high_traffic"},
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        return cls(**data)


class ExperimentRunner:
    """Main class for running and managing experiments."""
    
    def __init__(self, 
                 base_dir: Path = None,
                 use_cuda: bool = True,
                 n_workers: int = 1):
        # Set base_dir to experiments/experiment_results by default
        if base_dir is None:
            base_dir = Path(__file__).parent / "experiment_results"
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.n_workers = n_workers
        self.results = []
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup experiment logging."""
        log_file = self.base_dir / f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment with the given configuration."""
        logger.info(f"Starting experiment: {config.name}")
        logger.info(f"SAR Config: State={config.state}, Action={config.action}, Reward={config.reward}")
        
        # Create experiment directory
        exp_dir = self.base_dir / config.name / datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(exp_dir / "config.json", 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        results = {
            "config": config.to_dict(),
            "training": {},
            "evaluation": {},
            "metrics": {}
        }
        
        try:
            # Training phase
            if config.timesteps > 0:
                logger.info("Starting training phase...")
                training_results = self._train_model(config, exp_dir)
                results["training"] = training_results
                model_path = training_results.get("model_path")
            else:
                # Skip training, use existing model
                model_path = self._find_existing_model(config)
                logger.info(f"Using existing model: {model_path}")
            
            # Evaluation phase
            logger.info("Starting evaluation phase...")
            eval_results = self._evaluate_model(config, model_path, exp_dir)
            results["evaluation"] = eval_results
            
            # Calculate aggregate metrics
            results["metrics"] = self._calculate_metrics(eval_results)
            
            # Save results
            with open(exp_dir / "results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Experiment {config.name} completed successfully")
            
        except Exception as e:
            logger.error(f"Experiment {config.name} failed: {e}", exc_info=True)
            results["error"] = str(e)
        
        self.results.append(results)
        return results
    
    def _train_model(self, config: ExperimentConfig, exp_dir: Path) -> Dict[str, Any]:
        """Train a model with the given configuration."""
        import torch
        
        # Model naming
        model_name = f"{config.algorithm}_{config.reward}_{config.vsl_mode}"
        full_model_name = f"{model_name}_{config.state}_{config.action}"
        
        # Load SAR config and hyperparameters
        sar_config = load_sar_config(model_name)
        hyperparams = config.hyperparams or load_hyperparameters(model_name, config.algorithm)
        
        # Training settings
        train_sim_length = 3600 * 2  # 2 hours
        
        # Create training environments
        train_envs = SubprocVecEnv([
            lambda i=i: create_train_env_compat(
                i, model_name, train_sim_length, 10,
                config.reward, config.vsl_mode, config.state, config.action,
                sumo_port=8000 + i
            )
            for i in range(config.n_train_envs)
        ])
        
        # Create model
        model = DQN(
            "MlpPolicy",
            train_envs,
            verbose=1,
            tensorboard_log=str(exp_dir / "tensorboard"),
            device='cuda' if self.use_cuda else 'cpu',
            **hyperparams
        )
        
        # Train
        start_time = datetime.now()
        model.learn(
            total_timesteps=config.timesteps,
            progress_bar=True,
            reset_num_timesteps=False
        )
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save model
        model_path = exp_dir / f"{full_model_name}.zip"
        model.save(str(model_path))
        
        # Clean up
        train_envs.close()
        
        return {
            "model_path": str(model_path),
            "training_time": training_time,
            "timesteps": config.timesteps,
        }
    
    def _evaluate_model(self, 
                       config: ExperimentConfig, 
                       model_path: str,
                       exp_dir: Path) -> Dict[str, Any]:
        """Evaluate a trained model across different scenarios."""
        # Load model
        model = DQN.load(model_path, device='cuda' if self.use_cuda else 'cpu')
        
        eval_results = {}
        
        for scenario in config.eval_scenarios:
            logger.info(f"Evaluating scenario: {scenario['name']} (demand={scenario['demand']})")
            
            # Create evaluation environment
            eval_env = DummyVecEnv([
                lambda: create_eval_env_compat(
                    f"{config.algorithm}_{config.reward}",
                    3600 * 4,  # 4 hour evaluation
                    config.reward,
                    config.vsl_mode,
                    config.state,
                    config.action,
                    sumo_port=10000
                )
            ])
            
            # Modify environment to use specific demand
            eval_env.env_method("set_gen_car_distrib", [["uniform", scenario["demand"]]])
            
            # Evaluate
            episode_rewards = []
            episode_lengths = []
            episode_metrics = defaultdict(list)
            
            for ep in range(config.n_eval_episodes):
                obs = eval_env.reset()
                done = False
                episode_reward = 0
                episode_length = 0
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(action)
                    episode_reward += reward[0]
                    episode_length += 1
                    
                    # Collect metrics from info
                    if info and len(info) > 0:
                        for key in ['flow_downstream', 'avg_speed_before', 
                                   'occupancy', 'queue_length_upstream', 'collisions']:
                            if key in info[0]:
                                episode_metrics[key].append(info[0][key])
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            # Calculate statistics
            eval_results[scenario['name']] = {
                "demand": scenario['demand'],
                "rewards": {
                    "mean": float(np.mean(episode_rewards)),
                    "std": float(np.std(episode_rewards)),
                    "min": float(np.min(episode_rewards)),
                    "max": float(np.max(episode_rewards)),
                },
                "episode_lengths": {
                    "mean": float(np.mean(episode_lengths)),
                    "std": float(np.std(episode_lengths)),
                },
                "metrics": {
                    key: {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                    for key, values in episode_metrics.items()
                    if len(values) > 0
                }
            }
            
            eval_env.close()
        
        return eval_results
    
    def _calculate_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate aggregate metrics across all scenarios."""
        metrics = {}
        
        # Average reward across scenarios
        all_rewards = []
        for scenario_name, scenario_results in eval_results.items():
            all_rewards.append(scenario_results["rewards"]["mean"])
        
        metrics["avg_reward"] = float(np.mean(all_rewards))
        metrics["reward_stability"] = float(np.std(all_rewards))
        
        # Performance under different traffic conditions
        for scenario_name, scenario_results in eval_results.items():
            metrics[f"reward_{scenario_name}"] = scenario_results["rewards"]["mean"]
            
            # Traffic metrics
            if "metrics" in scenario_results:
                for metric_name in ["flow_downstream", "avg_speed_before", "queue_length_upstream"]:
                    if metric_name in scenario_results["metrics"]:
                        metrics[f"{metric_name}_{scenario_name}"] = \
                            scenario_results["metrics"][metric_name]["mean"]
        
        # Calculate robustness score (inverse of performance variance across scenarios)
        reward_values = [v for k, v in metrics.items() if k.startswith("reward_") and "stability" not in k]
        if len(reward_values) > 1:
            metrics["robustness_score"] = 1.0 / (1.0 + np.std(reward_values))
        
        return metrics
    
    def _find_existing_model(self, config: ExperimentConfig) -> str:
        """Find an existing trained model matching the configuration."""
        model_name = f"{config.algorithm}_{config.reward}_{config.vsl_mode}_{config.state}_{config.action}"
        
        # Search in standard model directory
        model_dir = Path(f"./rl_models/{model_name}/")
        if model_dir.exists():
            # Look for best model or final model
            best_model = model_dir / "best_model.zip"
            if best_model.exists():
                return str(best_model)
            
            final_model = model_dir / f"{model_name}_final.zip"
            if final_model.exists():
                return str(final_model)
            
            # Find any model file
            model_files = list(model_dir.glob("*.zip"))
            if model_files:
                return str(model_files[-1])
        
        raise FileNotFoundError(f"No existing model found for configuration: {model_name}")
    
    def run_batch(self, configs: List[ExperimentConfig], parallel: bool = False) -> List[Dict[str, Any]]:
        """Run multiple experiments in batch."""
        logger.info(f"Running batch of {len(configs)} experiments")
        
        if parallel and self.n_workers > 1:
            with mp.Pool(processes=self.n_workers) as pool:
                results = pool.map(self._run_single_wrapper, configs)
        else:
            results = []
            for config in configs:
                result = self.run_experiment(config)
                results.append(result)
        
        # Generate summary report
        self._generate_batch_report(results)
        
        return results
    
    def _run_single_wrapper(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Wrapper for multiprocessing."""
        return self.run_experiment(config)
    
    def _generate_batch_report(self, results: List[Dict[str, Any]]):
        """Generate a summary report for batch experiments."""
        report_path = self.base_dir / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = {
            "total_experiments": len(results),
            "successful": sum(1 for r in results if "error" not in r),
            "failed": sum(1 for r in results if "error" in r),
            "experiments": []
        }
        
        for result in results:
            exp_summary = {
                "name": result["config"]["name"],
                "state": result["config"]["state"],
                "action": result["config"]["action"],
                "reward": result["config"]["reward"],
                "status": "success" if "error" not in result else "failed",
            }
            
            if "metrics" in result:
                exp_summary["metrics"] = result["metrics"]
            
            if "error" in result:
                exp_summary["error"] = result["error"]
            
            summary["experiments"].append(exp_summary)
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Batch report saved to {report_path}")


def load_config_file(config_path: Path) -> List[ExperimentConfig]:
    """Load experiment configurations from YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    configs = []
    
    # Handle different config file formats
    if "experiments" in data:
        # List of experiments
        for exp_data in data["experiments"]:
            configs.append(ExperimentConfig.from_dict(exp_data))
    
    elif "base_config" in data and "variations" in data:
        # Base config with variations
        base = data["base_config"]
        
        for var in data["variations"]:
            config_data = base.copy()
            config_data.update(var)
            configs.append(ExperimentConfig.from_dict(config_data))
    
    else:
        # Single experiment
        configs.append(ExperimentConfig.from_dict(data))
    
    return configs


def run_single_experiment(config: ExperimentConfig, **kwargs) -> Dict[str, Any]:
    """Convenience function to run a single experiment."""
    # Extract base_dir if provided, otherwise use default
    base_dir = kwargs.pop('base_dir', None)
    runner = ExperimentRunner(base_dir=base_dir, **kwargs)
    return runner.run_experiment(config)


def run_batch_experiments(configs: List[ExperimentConfig], **kwargs) -> List[Dict[str, Any]]:
    """Convenience function to run batch experiments."""
    # Extract base_dir if provided, otherwise use default
    base_dir = kwargs.pop('base_dir', None)
    runner = ExperimentRunner(base_dir=base_dir, **kwargs)
    return runner.run_batch(configs, parallel=kwargs.get("parallel", False))


def run_from_config_file(config_path: str, **kwargs) -> List[Dict[str, Any]]:
    """Run experiments from a configuration file."""
    configs = load_config_file(Path(config_path))
    return run_batch_experiments(configs, **kwargs)


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(description="Run DRL-VSL experiments")
    
    # Experiment specification
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=str, help='Path to config file')
    group.add_argument('--single', action='store_true', help='Run single experiment')
    group.add_argument('--all-sar', action='store_true', help='Run all SAR combinations')
    
    # Single experiment options
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--state', type=str, default='full_metrics',
                       choices=['full_metrics', 'minimal'])
    parser.add_argument('--action', type=str, default='absolute_speed',
                       choices=['absolute_speed', 'relative_speed'])
    parser.add_argument('--reward', type=str, default='balanced',
                       choices=['mobility', 'safety', 'balanced'])
    
    # Training options
    parser.add_argument('--timesteps', type=int, default=100_000,
                       help='Training timesteps (0 to skip training)')
    parser.add_argument('--n-envs', type=int, default=2,
                       help='Number of training environments')
    
    # Evaluation options
    parser.add_argument('--n-eval', type=int, default=5,
                       help='Number of evaluation episodes per scenario')
    parser.add_argument('--scenarios', type=str, nargs='+',
                       default=['low', 'medium', 'high'],
                       help='Evaluation scenarios')
    
    # Execution options
    parser.add_argument('--parallel', action='store_true',
                       help='Run experiments in parallel')
    parser.add_argument('--n-workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: experiments/experiment_results)')
    
    args = parser.parse_args()
    
    # Check SUMO
    if 'SUMO_HOME' not in os.environ:
        logger.error("Please set SUMO_HOME environment variable")
        sys.exit(1)
    
    # Set base directory
    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        base_dir = Path(__file__).parent / "experiment_results"
    
    # Prepare scenarios
    scenario_map = {
        'low': {'demand': 2000, 'name': 'low_traffic'},
        'medium': {'demand': 3500, 'name': 'medium_traffic'},
        'high': {'demand': 5000, 'name': 'high_traffic'},
        'congested': {'demand': 6000, 'name': 'congested'},
    }
    
    eval_scenarios = [scenario_map.get(s, {'demand': 3000, 'name': s}) 
                     for s in args.scenarios]
    
    # Run experiments
    if args.config:
        # Run from config file
        results = run_from_config_file(
            args.config,
            base_dir=base_dir,
            n_workers=args.n_workers,
            parallel=args.parallel
        )
        
    elif args.single:
        # Run single experiment
        config = ExperimentConfig(
            name=args.name or f"{args.state}_{args.action}_{args.reward}",
            state=args.state,
            action=args.action,
            reward=args.reward,
            timesteps=args.timesteps,
            n_train_envs=args.n_envs,
            n_eval_episodes=args.n_eval,
            eval_scenarios=eval_scenarios
        )
        
        result = run_single_experiment(
            config,
            base_dir=base_dir
        )
        
        # Print summary
        if "metrics" in result:
            print("\nExperiment Results:")
            print(f"Average Reward: {result['metrics'].get('avg_reward', 'N/A')}")
            if isinstance(result['metrics'].get('avg_reward'), (int, float)):
                print(f"Average Reward: {result['metrics']['avg_reward']:.2f}")
            print(f"Robustness Score: {result['metrics'].get('robustness_score', 'N/A')}")
            if isinstance(result['metrics'].get('robustness_score'), (int, float)):
                print(f"Robustness Score: {result['metrics']['robustness_score']:.2f}")
    
    elif args.all_sar:
        # Run all SAR combinations
        configs = []
        
        states = ['full_metrics', 'minimal']
        actions = ['absolute_speed', 'relative_speed']
        rewards = ['mobility', 'safety', 'balanced']
        
        for state in states:
            for action in actions:
                for reward in rewards:
                    configs.append(ExperimentConfig(
                        name=f"{state}_{action}_{reward}",
                        state=state,
                        action=action,
                        reward=reward,
                        timesteps=args.timesteps,
                        n_train_envs=args.n_envs,
                        n_eval_episodes=args.n_eval,
                        eval_scenarios=eval_scenarios
                    ))
        
        results = run_batch_experiments(
            configs,
            base_dir=base_dir,
            n_workers=args.n_workers,
            parallel=args.parallel
        )
        
        # Print summary
        print(f"\nCompleted {len(results)} experiments")
        successful = sum(1 for r in results if "error" not in r)
        print(f"Successful: {successful}/{len(results)}")


if __name__ == "__main__":
    import torch
    main()