# dqn_reports.py
"""
Comprehensive reporting and visualization system for DQN-based Variable Speed Limit control.
Supports training analysis, testing evaluation, and hyperparameter tuning visualization.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set scientific plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class DQNReportsGenerator:
    """
    Generates comprehensive reports and visualizations for DQN-based VSL control experiments.
    Supports analysis of training sessions, testing evaluations, and hyperparameter tuning.
    """
    
    def __init__(self, base_results_dir: str = "rl_models", output_dir: str = "reports"):
        self.base_results_dir = Path(base_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Academic color scheme for consistency
        self.colors = {
            'primary': '#2E86AB',      # Blue
            'secondary': '#A23B72',    # Purple  
            'tertiary': '#F18F01',     # Orange
            'quaternary': '#C73E1D',   # Red
            'success': '#2F9B69',      # Green
            'neutral': '#6C757D'       # Gray
        }
        
        # Configure matplotlib for academic plots
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'axes.linewidth': 1.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.shadow': True
        })

    def generate_training_report(self, model_name: str, include_tb_logs: bool = True) -> Dict:
        """
        Generate comprehensive training analysis report.
        
        Args:
            model_name: Name of the trained model (e.g., "DQN_balanced_recommend")
            include_tb_logs: Whether to parse TensorBoard logs
            
        Returns:
            Dictionary containing analysis results and figure paths
        """
        print(f"Generating training report for {model_name}...")
        
        model_dir = self.base_results_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        report_dir = self.output_dir / f"{model_name}_training"
        report_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # 1. Learning curves from TensorBoard logs
        if include_tb_logs:
            tb_metrics = self._parse_tensorboard_logs(model_dir)
            if tb_metrics:
                results['learning_curves'] = self._plot_learning_curves(
                    tb_metrics, report_dir, model_name
                )
        
        # 2. Training progress from CSV logs  
        csv_metrics = self._parse_csv_logs(model_dir)
        if csv_metrics:
            results['training_progress'] = self._plot_training_progress(
                csv_metrics, report_dir, model_name
            )
        
        # 3. Environment interaction analysis
        env_metrics = self._parse_environment_logs(model_dir)
        if env_metrics:
            results['environment_analysis'] = self._plot_environment_metrics(
                env_metrics, report_dir, model_name
            )
        
        # 4. Generate training summary
        results['summary'] = self._generate_training_summary(
            tb_metrics, csv_metrics, env_metrics, report_dir, model_name
        )
        
        print(f"Training report saved to: {report_dir}")
        return results

    def generate_testing_report(self, model_name: str, baseline_comparison: bool = True) -> Dict:
        """
        Generate comprehensive testing evaluation report.
        
        Args:
            model_name: Name of the tested model
            baseline_comparison: Whether to include baseline (no-control) comparison
            
        Returns:
            Dictionary containing analysis results and figure paths
        """
        print(f"Generating testing report for {model_name}...")
        
        model_dir = self.base_results_dir / model_name
        report_dir = self.output_dir / f"{model_name}_testing"
        report_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # 1. Parse testing data
        test_data = self._parse_testing_data(model_dir)
        if not test_data:
            raise FileNotFoundError(f"No testing data found for {model_name}")
        
        # 2. Traffic performance analysis
        results['traffic_performance'] = self._plot_traffic_performance(
            test_data, report_dir, model_name
        )
        
        # 3. VSL control analysis
        results['vsl_analysis'] = self._plot_vsl_control_patterns(
            test_data, report_dir, model_name
        )
        
        # 4. Safety and efficiency metrics
        results['safety_metrics'] = self._plot_safety_efficiency_metrics(
            test_data, report_dir, model_name
        )
        
        # 5. Baseline comparison (if available)
        if baseline_comparison:
            baseline_data = self._load_baseline_data(model_dir)
            if baseline_data:
                results['baseline_comparison'] = self._plot_baseline_comparison(
                    test_data, baseline_data, report_dir, model_name
                )
        
        # 6. Generate testing summary
        results['summary'] = self._generate_testing_summary(
            test_data, report_dir, model_name
        )
        
        print(f"Testing report saved to: {report_dir}")
        return results

    def generate_hyperparameter_tuning_report(self, study_name: str, 
                                            algorithm: str = "DQN", 
                                            reward_function: str = "balanced",
                                            vsl_enforcement: str = "recommend") -> Dict:
        """
        Generate hyperparameter tuning analysis report.
        
        Args:
            study_name: Name of the Optuna study
            algorithm: RL algorithm used
            reward_function: Reward function used
            vsl_enforcement: VSL enforcement mode
            
        Returns:
            Dictionary containing analysis results and figure paths
        """
        print(f"Generating hyperparameter tuning report for {study_name}...")
        
        # Look for Optuna results
        optuna_dir = self.base_results_dir / "optuna_params"
        study_file = optuna_dir / f"optuna_params_{algorithm}_{reward_function}_{vsl_enforcement}.json"
        
        if not study_file.exists():
            raise FileNotFoundError(f"Optuna study file not found: {study_file}")
        
        report_dir = self.output_dir / f"{study_name}_hyperparams"
        report_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # 1. Load hyperparameter tuning data
        tuning_data = self._parse_optuna_results(study_file, optuna_dir)
        
        # 2. Parameter importance analysis
        results['parameter_importance'] = self._plot_parameter_importance(
            tuning_data, report_dir, study_name
        )
        
        # 3. Optimization history
        results['optimization_history'] = self._plot_optimization_history(
            tuning_data, report_dir, study_name
        )
        
        # 4. Parameter relationships
        results['parameter_relationships'] = self._plot_parameter_relationships(
            tuning_data, report_dir, study_name
        )
        
        # 5. Performance distribution
        results['performance_distribution'] = self._plot_performance_distribution(
            tuning_data, report_dir, study_name
        )
        
        # 6. Generate hyperparameter summary
        results['summary'] = self._generate_hyperparameter_summary(
            tuning_data, report_dir, study_name
        )
        
        print(f"Hyperparameter tuning report saved to: {report_dir}")
        return results

    def compare_models(self, model_names: List[str], comparison_type: str = "testing") -> Dict:
        """
        Generate comparative analysis between multiple models.
        
        Args:
            model_names: List of model names to compare
            comparison_type: Type of comparison ("testing", "training", "hyperparams")
            
        Returns:
            Dictionary containing comparison results and figure paths
        """
        print(f"Generating {comparison_type} comparison for models: {model_names}")
        
        report_dir = self.output_dir / f"comparison_{comparison_type}"
        report_dir.mkdir(exist_ok=True)
        
        results = {}
        
        if comparison_type == "testing":
            results = self._compare_testing_performance(model_names, report_dir)
        elif comparison_type == "training":
            results = self._compare_training_performance(model_names, report_dir)
        elif comparison_type == "hyperparams":
            results = self._compare_hyperparameter_studies(model_names, report_dir)
        
        print(f"Comparison report saved to: {report_dir}")
        return results

    # ==================== PARSING METHODS ====================
    
    def _parse_tensorboard_logs(self, model_dir: Path) -> Optional[Dict]:
        """Parse TensorBoard event files for training metrics."""
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            
            logs_dir = model_dir / "logs"
            if not logs_dir.exists():
                return None
            
            # Find the most recent tensorboard log directory
            tb_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
            if not tb_dirs:
                return None
            
            latest_tb_dir = max(tb_dirs, key=lambda x: x.stat().st_mtime)
            
            ea = EventAccumulator(str(latest_tb_dir))
            ea.Reload()
            
            metrics = {}
            
            # Extract common DQN metrics
            scalar_tags = ea.Tags()['scalars']
            for tag in scalar_tags:
                if any(keyword in tag.lower() for keyword in 
                       ['reward', 'loss', 'epsilon', 'learning_rate', 'episode']):
                    scalar_events = ea.Scalars(tag)
                    metrics[tag] = {
                        'steps': [event.step for event in scalar_events],
                        'values': [event.value for event in scalar_events],
                        'wall_time': [event.wall_time for event in scalar_events]
                    }
            
            return metrics
            
        except ImportError:
            print("TensorBoard not available for log parsing")
            return None
        except Exception as e:
            print(f"Error parsing TensorBoard logs: {e}")
            return None

    def _parse_csv_logs(self, model_dir: Path) -> Optional[pd.DataFrame]:
        """Parse CSV progress logs."""
        try:
            logs_dir = model_dir / "logs"
            csv_files = list(logs_dir.glob("progress.csv"))
            
            if not csv_files:
                return None
            
            # Use the most recent CSV file
            latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
            return pd.read_csv(latest_csv)
            
        except Exception as e:
            print(f"Error parsing CSV logs: {e}")
            return None

    def _parse_environment_logs(self, model_dir: Path) -> Optional[Dict]:
        """Parse environment-specific logs and detector data."""
        try:
            # Look for SUMO detector outputs
            detector_files = list(model_dir.glob("**/loop_detector_metrics.xml"))
            
            if not detector_files:
                return None
            
            # Parse the most recent detector file
            latest_detector = max(detector_files, key=lambda x: x.stat().st_mtime)
            return self._parse_detector_xml(latest_detector)
            
        except Exception as e:
            print(f"Error parsing environment logs: {e}")
            return None

    def _parse_detector_xml(self, xml_file: Path) -> Dict:
        """Parse SUMO detector XML output."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            data = {
                'timesteps': [],
                'flows': [],
                'occupancies': [],
                'speeds': [],
                'detector_ids': []
            }
            
            for interval in root.findall('.//interval'):
                timestep = float(interval.get('begin', 0))
                detector_id = interval.get('id', '')
                flow = float(interval.get('flow', 0))
                occupancy = float(interval.get('occupancy', 0))
                speed = float(interval.get('speed', 0))
                
                data['timesteps'].append(timestep)
                data['flows'].append(flow)
                data['occupancies'].append(occupancy)
                data['speeds'].append(speed)
                data['detector_ids'].append(detector_id)
            
            return data
            
        except Exception as e:
            print(f"Error parsing detector XML: {e}")
            return {}

    def _parse_testing_data(self, model_dir: Path) -> Optional[Dict]:
        """Parse testing session data."""
        # This would parse test results, which might be in various formats
        # depending on how testing data is saved
        try:
            # Look for test results JSON or CSV files
            test_files = list(model_dir.glob("test_results.*"))
            
            if not test_files:
                # Try to find detector data from testing
                return self._parse_environment_logs(model_dir)
            
            # Parse the test results file
            test_file = test_files[0]
            if test_file.suffix == '.json':
                with open(test_file, 'r') as f:
                    return json.load(f)
            elif test_file.suffix == '.csv':
                df = pd.read_csv(test_file)
                return df.to_dict('list')
            
        except Exception as e:
            print(f"Error parsing testing data: {e}")
            
        return None

    def _parse_optuna_results(self, study_file: Path, optuna_dir: Path) -> Dict:
        """Parse Optuna hyperparameter tuning results."""
        try:
            with open(study_file, 'r') as f:
                best_params = json.load(f)
            
            # Look for detailed study results
            study_db_file = optuna_dir / "study.db"
            detailed_results = None
            
            if study_db_file.exists():
                try:
                    import optuna
                    study = optuna.load_study(
                        study_name="dqn_vsl_study", 
                        storage=f"sqlite:///{study_db_file}"
                    )
                    
                    trials_data = []
                    for trial in study.trials:
                        trial_data = {
                            'number': trial.number,
                            'value': trial.value,
                            'state': trial.state.name,
                            'params': trial.params,
                            'duration': trial.duration.total_seconds() if trial.duration else None
                        }
                        trials_data.append(trial_data)
                    
                    detailed_results = {
                        'trials': trials_data,
                        'best_trial': {
                            'number': study.best_trial.number,
                            'value': study.best_trial.value,
                            'params': study.best_trial.params
                        }
                    }
                    
                except ImportError:
                    print("Optuna not available for detailed study parsing")
                
            return {
                'best_params': best_params,
                'detailed_results': detailed_results
            }
            
        except Exception as e:
            print(f"Error parsing Optuna results: {e}")
            return {}

    # ==================== PLOTTING METHODS ====================
    
    def _plot_learning_curves(self, tb_metrics: Dict, output_dir: Path, model_name: str) -> str:
        """Plot learning curves from TensorBoard data."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Learning Curves - {model_name}', fontsize=16, fontweight='bold')
        
        # Plot reward progression
        if any('reward' in tag.lower() for tag in tb_metrics.keys()):
            ax = axes[0, 0]
            for tag, data in tb_metrics.items():
                if 'reward' in tag.lower():
                    ax.plot(data['steps'], data['values'], label=tag.replace('/', ' '), linewidth=2)
            ax.set_title('Episode Rewards')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot loss progression
        if any('loss' in tag.lower() for tag in tb_metrics.keys()):
            ax = axes[0, 1]
            for tag, data in tb_metrics.items():
                if 'loss' in tag.lower():
                    ax.plot(data['steps'], data['values'], label=tag.replace('/', ' '), linewidth=2)
            ax.set_title('Training Loss')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot epsilon decay
        if any('epsilon' in tag.lower() for tag in tb_metrics.keys()):
            ax = axes[1, 0]
            for tag, data in tb_metrics.items():
                if 'epsilon' in tag.lower():
                    ax.plot(data['steps'], data['values'], label=tag.replace('/', ' '), linewidth=2)
            ax.set_title('Exploration Rate (Epsilon)')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Epsilon')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot learning rate
        if any('learning_rate' in tag.lower() for tag in tb_metrics.keys()):
            ax = axes[1, 1]
            for tag, data in tb_metrics.items():
                if 'learning_rate' in tag.lower():
                    ax.plot(data['steps'], data['values'], label=tag.replace('/', ' '), linewidth=2)
            ax.set_title('Learning Rate')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Learning Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = output_dir / f"{model_name}_learning_curves.png"
        plt.savefig(output_file)
        plt.close()
        
        return str(output_file)

    def _plot_training_progress(self, csv_data: pd.DataFrame, output_dir: Path, model_name: str) -> str:
        """Plot training progress from CSV logs."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Training Progress - {model_name}', fontsize=16, fontweight='bold')
        
        # Episode length progression
        if 'ep_len_mean' in csv_data.columns:
            axes[0, 0].plot(csv_data.index, csv_data['ep_len_mean'], 
                           color=self.colors['primary'], linewidth=2)
            axes[0, 0].set_title('Episode Length')
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('Mean Episode Length')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Reward progression with confidence interval
        if 'ep_rew_mean' in csv_data.columns:
            axes[0, 1].plot(csv_data.index, csv_data['ep_rew_mean'], 
                           color=self.colors['secondary'], linewidth=2, label='Mean')
            
            if 'ep_rew_std' in csv_data.columns:
                mean_reward = csv_data['ep_rew_mean']
                std_reward = csv_data['ep_rew_std']
                axes[0, 1].fill_between(csv_data.index, 
                                       mean_reward - std_reward,
                                       mean_reward + std_reward,
                                       alpha=0.3, color=self.colors['secondary'])
            
            axes[0, 1].set_title('Episode Rewards')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Training time progression
        if 'time_elapsed' in csv_data.columns:
            axes[1, 0].plot(csv_data.index, csv_data['time_elapsed']/3600,  # Convert to hours
                           color=self.colors['tertiary'], linewidth=2)
            axes[1, 0].set_title('Training Time')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Time Elapsed (hours)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # FPS (Frames per second)
        if 'fps' in csv_data.columns:
            axes[1, 1].plot(csv_data.index, csv_data['fps'], 
                           color=self.colors['quaternary'], linewidth=2)
            axes[1, 1].set_title('Training Speed')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('FPS')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = output_dir / f"{model_name}_training_progress.png"
        plt.savefig(output_file)
        plt.close()
        
        return str(output_file)

    def _plot_environment_metrics(self, env_data: Dict, output_dir: Path, model_name: str) -> str:
        """Plot environment-specific metrics during training."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Environment Metrics During Training - {model_name}', fontsize=16, fontweight='bold')
        
        timesteps = np.array(env_data.get('timesteps', []))
        
        # Traffic flow
        flows = np.array(env_data.get('flows', []))
        if len(flows) > 0:
            axes[0, 0].plot(timesteps, flows, color=self.colors['primary'], linewidth=1.5)
            axes[0, 0].set_title('Traffic Flow')
            axes[0, 0].set_xlabel('Simulation Time (s)')
            axes[0, 0].set_ylabel('Flow (veh/h)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Occupancy
        occupancies = np.array(env_data.get('occupancies', []))
        if len(occupancies) > 0:
            axes[0, 1].plot(timesteps, occupancies, color=self.colors['secondary'], linewidth=1.5)
            axes[0, 1].set_title('Lane Occupancy')
            axes[0, 1].set_xlabel('Simulation Time (s)')
            axes[0, 1].set_ylabel('Occupancy (%)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Speed distribution
        speeds = np.array(env_data.get('speeds', []))
        if len(speeds) > 0:
            axes[1, 0].hist(speeds[speeds > 0], bins=30, color=self.colors['tertiary'], 
                           alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Speed Distribution')
            axes[1, 0].set_xlabel('Speed (km/h)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Flow vs Occupancy scatter
        if len(flows) > 0 and len(occupancies) > 0:
            valid_idx = (flows > 0) & (occupancies > 0)
            if np.any(valid_idx):
                axes[1, 1].scatter(occupancies[valid_idx], flows[valid_idx], 
                                 color=self.colors['quaternary'], alpha=0.6, s=20)
                axes[1, 1].set_title('Flow-Density Relationship')
                axes[1, 1].set_xlabel('Occupancy (%)')
                axes[1, 1].set_ylabel('Flow (veh/h)')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = output_dir / f"{model_name}_environment_metrics.png"
        plt.savefig(output_file)
        plt.close()
        
        return str(output_file)

    def _plot_traffic_performance(self, test_data: Dict, output_dir: Path, model_name: str) -> str:
        """Plot traffic performance metrics from testing."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Traffic Performance Analysis - {model_name}', fontsize=16, fontweight='bold')
        
        timesteps = np.array(test_data.get('timesteps', []))
        
        # Traffic flow time series
        flows = np.array(test_data.get('flows', []))
        if len(flows) > 0:
            axes[0, 0].plot(timesteps/3600, flows, color=self.colors['primary'], linewidth=2)
            axes[0, 0].set_title('Traffic Flow Over Time')
            axes[0, 0].set_xlabel('Time (hours)')
            axes[0, 0].set_ylabel('Flow (veh/h)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Speed time series
        speeds = np.array(test_data.get('speeds', []))
        if len(speeds) > 0:
            axes[0, 1].plot(timesteps/3600, speeds, color=self.colors['secondary'], linewidth=2)
            axes[0, 1].set_title('Average Speed Over Time')
            axes[0, 1].set_xlabel('Time (hours)')
            axes[0, 1].set_ylabel('Speed (km/h)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Occupancy time series
        occupancies = np.array(test_data.get('occupancies', []))
        if len(occupancies) > 0:
            axes[0, 2].plot(timesteps/3600, occupancies, color=self.colors['tertiary'], linewidth=2)
            axes[0, 2].set_title('Lane Occupancy Over Time')
            axes[0, 2].set_xlabel('Time (hours)')
            axes[0, 2].set_ylabel('Occupancy (%)')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Performance distributions
        if len(flows) > 0:
            axes[1, 0].hist(flows, bins=30, color=self.colors['primary'], 
                           alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(np.mean(flows), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(flows):.0f}')
            axes[1, 0].set_title('Flow Distribution')
            axes[1, 0].set_xlabel('Flow (veh/h)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        if len(speeds) > 0 and np.any(speeds > 0):
            valid_speeds = speeds[speeds > 0]
            axes[1, 1].hist(valid_speeds, bins=30, color=self.colors['secondary'], 
                           alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(np.mean(valid_speeds), color='red', linestyle='--',
                              label=f'Mean: {np.mean(valid_speeds):.1f}')
            axes[1, 1].set_title('Speed Distribution')
            axes[1, 1].set_xlabel('Speed (km/h)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        if len(occupancies) > 0:
            axes[1, 2].hist(occupancies, bins=30, color=self.colors['tertiary'], 
                           alpha=0.7, edgecolor='black')
            axes[1, 2].axvline(np.mean(occupancies), color='red', linestyle='--',
                              label=f'Mean: {np.mean(occupancies):.1f}%')
            axes[1, 2].set_title('Occupancy Distribution')
            axes[1, 2].set_xlabel('Occupancy (%)')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = output_dir / f"{model_name}_traffic_performance.png"
        plt.savefig(output_file)
        plt.close()
        
        return str(output_file)

    def _plot_vsl_control_patterns(self, test_data: Dict, output_dir: Path, model_name: str) -> str:
        """Plot VSL control patterns and effectiveness."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'VSL Control Analysis - {model_name}', fontsize=16, fontweight='bold')
        
        # This would need to be extracted from your test data
        # For now, creating placeholder structure
        timesteps = np.array(test_data.get('timesteps', []))
        
        # VSL commands over time (placeholder - would need actual VSL data)
        if 'vsl_commands' in test_data:
            vsl_commands = np.array(test_data['vsl_commands'])
            axes[0, 0].plot(timesteps/3600, vsl_commands, 
                           color=self.colors['primary'], linewidth=2, marker='o', markersize=3)
            axes[0, 0].set_title('VSL Commands Over Time')
            axes[0, 0].set_xlabel('Time (hours)')
            axes[0, 0].set_ylabel('Speed Limit (km/h)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # VSL activation frequency
        if 'vsl_commands' in test_data:
            vsl_commands = np.array(test_data['vsl_commands'])
            unique_limits, counts = np.unique(vsl_commands, return_counts=True)
            axes[0, 1].bar(unique_limits, counts, color=self.colors['secondary'], alpha=0.7)
            axes[0, 1].set_title('VSL Speed Limit Usage')
            axes[0, 1].set_xlabel('Speed Limit (km/h)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Effectiveness: Flow vs VSL
        if 'vsl_commands' in test_data and 'flows' in test_data:
            flows = np.array(test_data['flows'])
            vsl_commands = np.array(test_data['vsl_commands'])
            
            # Create scatter plot
            axes[1, 0].scatter(vsl_commands, flows, alpha=0.6, 
                              color=self.colors['tertiary'], s=20)
            axes[1, 0].set_title('Flow Response to VSL')
            axes[1, 0].set_xlabel('VSL Speed Limit (km/h)')
            axes[1, 0].set_ylabel('Flow (veh/h)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Control action distribution
        if 'actions' in test_data:
            actions = np.array(test_data['actions'])
            action_labels = ['Decrease -10', 'Decrease -5', 'Maintain', 'Increase +5', 'Increase +10']
            unique_actions, counts = np.unique(actions, return_counts=True)
            
            axes[1, 1].pie(counts, labels=[action_labels[i] for i in unique_actions], 
                          autopct='%1.1f%%', startangle=90, 
                          colors=[self.colors[key] for key in ['quaternary', 'secondary', 'neutral', 'primary', 'success']])
            axes[1, 1].set_title('Action Distribution')
        
        plt.tight_layout()
        
        output_file = output_dir / f"{model_name}_vsl_control.png"
        plt.savefig(output_file)
        plt.close()
        
        return str(output_file)

    def _plot_safety_efficiency_metrics(self, test_data: Dict, output_dir: Path, model_name: str) -> str:
        """Plot safety and efficiency metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Safety & Efficiency Metrics - {model_name}', fontsize=16, fontweight='bold')
        
        # Calculate speed variance as safety indicator
        speeds = np.array(test_data.get('speeds', []))
        if len(speeds) > 0:
            # Rolling speed variance
            window_size = 10
            speed_variance = pd.Series(speeds).rolling(window=window_size).var()
            
            axes[0, 0].plot(speed_variance.index, speed_variance.values, 
                           color=self.colors['quaternary'], linewidth=2)
            axes[0, 0].set_title('Speed Variance (Safety Indicator)')
            axes[0, 0].set_xlabel('Time Steps')
            axes[0, 0].set_ylabel('Speed Variance')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Calculate throughput efficiency
        flows = np.array(test_data.get('flows', []))
        if len(flows) > 0:
            # Cumulative throughput
            cumulative_throughput = np.cumsum(flows) / 3600  # Convert to vehicles
            
            axes[0, 1].plot(cumulative_throughput, color=self.colors['success'], linewidth=2)
            axes[0, 1].set_title('Cumulative Throughput')
            axes[0, 1].set_xlabel('Time Steps')
            axes[0, 1].set_ylabel('Total Vehicles Passed')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Queue length analysis (if available)
        if 'queue_lengths' in test_data:
            queue_lengths = np.array(test_data['queue_lengths'])
            axes[1, 0].plot(queue_lengths, color=self.colors['secondary'], linewidth=2)
            axes[1, 0].set_title('Queue Length Over Time')
            axes[1, 0].set_xlabel('Time Steps')
            axes[1, 0].set_ylabel('Queue Length (vehicles)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Efficiency vs Safety trade-off
        if len(speeds) > 0 and len(flows) > 0:
            # Use rolling averages for smoother visualization
            window_size = 10
            avg_speeds = pd.Series(speeds).rolling(window=window_size).mean()
            avg_flows = pd.Series(flows).rolling(window=window_size).mean()
            
            valid_idx = ~(np.isnan(avg_speeds) | np.isnan(avg_flows))
            
            axes[1, 1].scatter(avg_speeds[valid_idx], avg_flows[valid_idx], 
                              alpha=0.6, color=self.colors['primary'], s=20)
            axes[1, 1].set_title('Efficiency vs Safety Trade-off')
            axes[1, 1].set_xlabel('Average Speed (km/h)')
            axes[1, 1].set_ylabel('Flow (veh/h)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = output_dir / f"{model_name}_safety_efficiency.png"
        plt.savefig(output_file)
        plt.close()
        
        return str(output_file)

    def _plot_parameter_importance(self, tuning_data: Dict, output_dir: Path, study_name: str) -> str:
        """Plot hyperparameter importance analysis."""
        if not tuning_data.get('detailed_results'):
            return ""
        
        trials = tuning_data['detailed_results']['trials']
        
        # Extract parameter data
        param_names = set()
        for trial in trials:
            if trial['state'] == 'COMPLETE':
                param_names.update(trial['params'].keys())
        
        param_names = list(param_names)
        
        # Calculate parameter importance using correlation with objective value
        param_importance = {}
        
        for param in param_names:
            param_values = []
            objective_values = []
            
            for trial in trials:
                if trial['state'] == 'COMPLETE' and param in trial['params']:
                    param_val = trial['params'][param]
                    # Convert string parameters to numeric for correlation
                    if isinstance(param_val, str):
                        if param == 'net_arch_str':
                            # Convert network architecture to complexity measure
                            layers = [int(x) for x in param_val.split(',')]
                            param_val = sum(layers)  # Total neurons as complexity measure
                        else:
                            continue  # Skip non-numeric string parameters
                    
                    param_values.append(param_val)
                    objective_values.append(trial['value'])
            
            if len(param_values) > 1:
                correlation = np.corrcoef(param_values, objective_values)[0, 1]
                param_importance[param] = abs(correlation)
        
        # Plot parameter importance
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if param_importance:
            params = list(param_importance.keys())
            importance_values = list(param_importance.values())
            
            bars = ax.barh(params, importance_values, color=self.colors['primary'], alpha=0.7)
            ax.set_xlabel('Importance (Absolute Correlation)')
            ax.set_title(f'Hyperparameter Importance - {study_name}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, importance in zip(bars, importance_values):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{importance:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        output_file = output_dir / f"{study_name}_parameter_importance.png"
        plt.savefig(output_file)
        plt.close()
        
        return str(output_file)

    def _plot_optimization_history(self, tuning_data: Dict, output_dir: Path, study_name: str) -> str:
        """Plot optimization history and convergence."""
        if not tuning_data.get('detailed_results'):
            return ""
        
        trials = tuning_data['detailed_results']['trials']
        
        # Extract trial numbers and objective values
        trial_numbers = []
        objective_values = []
        best_values = []
        
        current_best = float('-inf')
        
        for trial in trials:
            if trial['state'] == 'COMPLETE':
                trial_numbers.append(trial['number'])
                obj_val = trial['value']
                objective_values.append(obj_val)
                
                if obj_val > current_best:
                    current_best = obj_val
                best_values.append(current_best)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Optimization History - {study_name}', fontsize=16, fontweight='bold')
        
        # Plot objective values over trials
        ax1.scatter(trial_numbers, objective_values, alpha=0.6, 
                   color=self.colors['primary'], s=30, label='Trial Values')
        ax1.plot(trial_numbers, best_values, color=self.colors['quaternary'], 
                linewidth=3, label='Best Value')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Objective Value Progression')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot convergence
        improvement_gaps = []
        for i in range(1, len(best_values)):
            gap = best_values[i] - best_values[i-1]
            improvement_gaps.append(gap)
        
        if improvement_gaps:
            ax2.plot(trial_numbers[1:], improvement_gaps, color=self.colors['secondary'], 
                    linewidth=2, marker='o', markersize=4)
            ax2.set_xlabel('Trial Number')
            ax2.set_ylabel('Improvement')
            ax2.set_title('Convergence Analysis')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = output_dir / f"{study_name}_optimization_history.png"
        plt.savefig(output_file)
        plt.close()
        
        return str(output_file)

    def _plot_parameter_relationships(self, tuning_data: Dict, output_dir: Path, study_name: str) -> str:
        """Plot relationships between hyperparameters."""
        if not tuning_data.get('detailed_results'):
            return ""
        
        trials = tuning_data['detailed_results']['trials']
        
        # Extract completed trials data
        completed_trials = [t for t in trials if t['state'] == 'COMPLETE']
        
        if len(completed_trials) < 10:  # Need sufficient data for visualization
            return ""
        
        # Create parameter matrix
        param_data = {}
        objective_values = []
        
        for trial in completed_trials:
            objective_values.append(trial['value'])
            for param, value in trial['params'].items():
                if param not in param_data:
                    param_data[param] = []
                
                # Convert parameters to numeric values
                if isinstance(value, str):
                    if param == 'net_arch_str':
                        layers = [int(x) for x in value.split(',')]
                        numeric_value = sum(layers)  # Total neurons
                    else:
                        continue
                else:
                    numeric_value = value
                
                param_data[param].append(numeric_value)
        
        # Select top parameters for visualization
        numeric_params = {k: v for k, v in param_data.items() if len(v) == len(completed_trials)}
        
        if len(numeric_params) < 2:
            return ""
        
        param_names = list(numeric_params.keys())[:4]  # Limit to 4 parameters for clarity
        
        # Create correlation matrix
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Parameter Relationships - {study_name}', fontsize=16, fontweight='bold')
        
        # Plot parameter vs objective for top parameters
        for i, param in enumerate(param_names[:4]):
            ax = axes[i//2, i%2]
            
            param_values = numeric_params[param]
            
            # Color points by objective value
            scatter = ax.scatter(param_values, objective_values, 
                               c=objective_values, cmap='viridis', 
                               alpha=0.7, s=50)
            
            ax.set_xlabel(param.replace('_', ' ').title())
            ax.set_ylabel('Objective Value')
            ax.set_title(f'{param.replace("_", " ").title()} vs Performance')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Objective Value')
        
        plt.tight_layout()
        
        output_file = output_dir / f"{study_name}_parameter_relationships.png"
        plt.savefig(output_file)
        plt.close()
        
        return str(output_file)

    def _plot_performance_distribution(self, tuning_data: Dict, output_dir: Path, study_name: str) -> str:
        """Plot performance distribution across trials."""
        if not tuning_data.get('detailed_results'):
            return ""
        
        trials = tuning_data['detailed_results']['trials']
        
        # Extract objective values
        objective_values = [t['value'] for t in trials if t['state'] == 'COMPLETE']
        
        if not objective_values:
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Performance Distribution - {study_name}', fontsize=16, fontweight='bold')
        
        # Histogram
        ax1.hist(objective_values, bins=20, color=self.colors['primary'], 
                alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(objective_values), color=self.colors['quaternary'], 
                   linestyle='--', linewidth=2, label=f'Mean: {np.mean(objective_values):.3f}')
        ax1.axvline(np.median(objective_values), color=self.colors['secondary'], 
                   linestyle='--', linewidth=2, label=f'Median: {np.median(objective_values):.3f}')
        ax1.set_xlabel('Objective Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Performance Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(objective_values, patch_artist=True,
                   boxprops=dict(facecolor=self.colors['primary'], alpha=0.7),
                   medianprops=dict(color=self.colors['quaternary'], linewidth=2))
        ax2.set_ylabel('Objective Value')
        ax2.set_title('Performance Statistics')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""
        Best: {max(objective_values):.3f}
        Worst: {min(objective_values):.3f}
        Std: {np.std(objective_values):.3f}
        """
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        output_file = output_dir / f"{study_name}_performance_distribution.png"
        plt.savefig(output_file)
        plt.close()
        
        return str(output_file)

    # ==================== SUMMARY METHODS ====================
    
    def _generate_training_summary(self, tb_metrics: Dict, csv_data: pd.DataFrame, 
                                 env_data: Dict, output_dir: Path, model_name: str) -> str:
        """Generate comprehensive training summary report."""
        summary = {
            'model_name': model_name,
            'generation_time': datetime.now().isoformat(),
            'training_metrics': {},
            'environment_metrics': {},
            'recommendations': []
        }
        
        # Training metrics summary
        if csv_data is not None and not csv_data.empty:
            summary['training_metrics'] = {
                'total_timesteps': len(csv_data),
                'final_reward_mean': float(csv_data['ep_rew_mean'].iloc[-1]) if 'ep_rew_mean' in csv_data.columns else None,
                'best_reward': float(csv_data['ep_rew_mean'].max()) if 'ep_rew_mean' in csv_data.columns else None,
                'training_stability': float(csv_data['ep_rew_mean'].std()) if 'ep_rew_mean' in csv_data.columns else None,
                'convergence_achieved': self._assess_convergence(csv_data)
            }
        
        # Environment metrics summary
        if env_data:
            flows = np.array(env_data.get('flows', []))
            speeds = np.array(env_data.get('speeds', []))
            occupancies = np.array(env_data.get('occupancies', []))
            
            if len(flows) > 0:
                summary['environment_metrics'] = {
                    'avg_flow': float(np.mean(flows)),
                    'avg_speed': float(np.mean(speeds[speeds > 0])) if len(speeds[speeds > 0]) > 0 else None,
                    'avg_occupancy': float(np.mean(occupancies)),
                    'flow_stability': float(np.std(flows))
                }
        
        # Generate recommendations
        summary['recommendations'] = self._generate_training_recommendations(
            summary['training_metrics'], summary['environment_metrics']
        )
        
        # Save summary
        summary_file = output_dir / f"{model_name}_training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        return str(summary_file)

    def _generate_testing_summary(self, test_data: Dict, output_dir: Path, model_name: str) -> str:
        """Generate testing summary report."""
        summary = {
            'model_name': model_name,
            'generation_time': datetime.now().isoformat(),
            'performance_metrics': {},
            'traffic_metrics': {},
            'recommendations': []
        }
        
        # Calculate performance metrics
        flows = np.array(test_data.get('flows', []))
        speeds = np.array(test_data.get('speeds', []))
        occupancies = np.array(test_data.get('occupancies', []))
        
        if len(flows) > 0:
            summary['performance_metrics'] = {
                'total_throughput': float(np.sum(flows)),
                'avg_flow': float(np.mean(flows)),
                'flow_efficiency': float(np.mean(flows) / np.max(flows)) if np.max(flows) > 0 else 0,
                'flow_stability': float(1 / (1 + np.std(flows))) # Inverse of coefficient of variation
            }
        
        if len(speeds) > 0:
            valid_speeds = speeds[speeds > 0]
            if len(valid_speeds) > 0:
                summary['traffic_metrics'] = {
                    'avg_speed': float(np.mean(valid_speeds)),
                    'speed_stability': float(1 / (1 + np.std(valid_speeds))),
                    'min_speed': float(np.min(valid_speeds)),
                    'max_speed': float(np.max(valid_speeds))
                }
        
        # Generate recommendations
        summary['recommendations'] = self._generate_testing_recommendations(
            summary['performance_metrics'], summary['traffic_metrics']
        )
        
        # Save summary
        summary_file = output_dir / f"{model_name}_testing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        return str(summary_file)

    def _generate_hyperparameter_summary(self, tuning_data: Dict, output_dir: Path, study_name: str) -> str:
        """Generate hyperparameter tuning summary."""
        summary = {
            'study_name': study_name,
            'generation_time': datetime.now().isoformat(),
            'best_parameters': tuning_data.get('best_params', {}),
            'optimization_metrics': {},
            'recommendations': []
        }
        
        if tuning_data.get('detailed_results'):
            trials = tuning_data['detailed_results']['trials']
            completed_trials = [t for t in trials if t['state'] == 'COMPLETE']
            
            if completed_trials:
                objective_values = [t['value'] for t in completed_trials]
                
                summary['optimization_metrics'] = {
                    'total_trials': len(trials),
                    'completed_trials': len(completed_trials),
                    'best_value': float(max(objective_values)),
                    'worst_value': float(min(objective_values)),
                    'mean_value': float(np.mean(objective_values)),
                    'std_value': float(np.std(objective_values)),
                    'improvement_ratio': float((max(objective_values) - min(objective_values)) / abs(min(objective_values))) if min(objective_values) != 0 else 0
                }
        
        # Generate recommendations
        summary['recommendations'] = self._generate_hyperparameter_recommendations(
            summary['optimization_metrics'], summary['best_parameters']
        )
        
        # Save summary
        summary_file = output_dir / f"{study_name}_hyperparameter_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        return str(summary_file)

    # ==================== UTILITY METHODS ====================
    
    def _assess_convergence(self, csv_data: pd.DataFrame) -> bool:
        """Assess if training has converged based on reward stability."""
        if 'ep_rew_mean' not in csv_data.columns or len(csv_data) < 50:
            return False
        
        # Check last 20% of training for stability
        last_portion = int(len(csv_data) * 0.2)
        recent_rewards = csv_data['ep_rew_mean'].iloc[-last_portion:]
        
        # Simple convergence check: coefficient of variation < 0.1
        cv = recent_rewards.std() / abs(recent_rewards.mean()) if recent_rewards.mean() != 0 else float('inf')
        return cv < 0.1

    def _generate_training_recommendations(self, training_metrics: Dict, env_metrics: Dict) -> List[str]:
        """Generate training-specific recommendations."""
        recommendations = []
        
        if training_metrics.get('training_stability', 0) > 5:
            recommendations.append("High reward variance detected. Consider reducing learning rate or increasing batch size.")
        
        if not training_metrics.get('convergence_achieved', False):
            recommendations.append("Training may not have converged. Consider increasing total timesteps.")
        
        if training_metrics.get('final_reward_mean', 0) < 0:
            recommendations.append("Negative final reward suggests poor policy. Review reward function design.")
        
        return recommendations

    def _generate_testing_recommendations(self, performance_metrics: Dict, traffic_metrics: Dict) -> List[str]:
        """Generate testing-specific recommendations."""
        recommendations = []
        
        if performance_metrics.get('flow_efficiency', 0) < 0.7:
            recommendations.append("Low flow efficiency. Consider retraining with different reward weights.")
        
        if traffic_metrics.get('speed_stability', 0) < 0.8:
            recommendations.append("High speed variance detected. VSL control may be too aggressive.")
        
        return recommendations

    def _generate_hyperparameter_recommendations(self, opt_metrics: Dict, best_params: Dict) -> List[str]:
        """Generate hyperparameter tuning recommendations."""
        recommendations = []
        
        if opt_metrics.get('improvement_ratio', 0) < 0.1:
            recommendations.append("Low improvement ratio suggests hyperparameters may not significantly impact performance.")
        
        if opt_metrics.get('completed_trials', 0) < 30:
            recommendations.append("Consider running more trials for better hyperparameter space exploration.")
        
        return recommendations

    def _load_baseline_data(self, model_dir: Path) -> Optional[Dict]:
        """Load baseline (no-control) data for comparison."""
        baseline_file = model_dir / "baseline_results.json"
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                return json.load(f)
        return None

    def _plot_baseline_comparison(self, test_data: Dict, baseline_data: Dict, 
                                output_dir: Path, model_name: str) -> str:
        """Plot comparison with baseline performance."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Baseline Comparison - {model_name}', fontsize=16, fontweight='bold')
        
        # Compare key metrics
        metrics = ['flows', 'speeds', 'occupancies']
        metric_labels = ['Flow (veh/h)', 'Speed (km/h)', 'Occupancy (%)']
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['tertiary']]
        
        for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
            if i >= 3:  # Only plot first 3 metrics
                break
                
            ax = axes[i//2, i%2]
            
            test_values = np.array(test_data.get(metric, []))
            baseline_values = np.array(baseline_data.get(metric, []))
            
            if len(test_values) > 0 and len(baseline_values) > 0:
                ax.hist(baseline_values, bins=20, alpha=0.6, label='Baseline', 
                       color='gray', density=True)
                ax.hist(test_values, bins=20, alpha=0.7, label='DQN Control', 
                       color=color, density=True)
                
                ax.set_xlabel(label)
                ax.set_ylabel('Density')
                ax.set_title(f'{metric.title()} Comparison')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Summary comparison
        ax = axes[1, 1]
        
        # Calculate improvement percentages
        improvements = {}
        for metric in metrics:
            test_values = np.array(test_data.get(metric, []))
            baseline_values = np.array(baseline_data.get(metric, []))
            
            if len(test_values) > 0 and len(baseline_values) > 0:
                test_mean = np.mean(test_values)
                baseline_mean = np.mean(baseline_values)
                improvement = ((test_mean - baseline_mean) / baseline_mean) * 100
                improvements[metric] = improvement
        
        if improvements:
            metric_names = list(improvements.keys())
            improvement_values = list(improvements.values())
            
            bars = ax.bar(metric_names, improvement_values, 
                         color=[self.colors['success'] if x > 0 else self.colors['quaternary'] 
                               for x in improvement_values])
            
            ax.set_ylabel('Improvement (%)')
            ax.set_title('Performance Improvement vs Baseline')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            
            # Add value labels on bars
            for bar, improvement in zip(bars, improvement_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5 if height > 0 else height - 0.5,
                       f'{improvement:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        
        output_file = output_dir / f"{model_name}_baseline_comparison.png"
        plt.savefig(output_file)
        plt.close()
        
        return str(output_file)

    # ==================== COMPARISON METHODS ====================
    
    def _compare_testing_performance(self, model_names: List[str], output_dir: Path) -> Dict:
        """Compare testing performance across multiple models."""
        # Implementation for model comparison
        pass

    def _compare_training_performance(self, model_names: List[str], output_dir: Path) -> Dict:
        """Compare training performance across multiple models."""
        # Implementation for training comparison
        pass

    def _compare_hyperparameter_studies(self, study_names: List[str], output_dir: Path) -> Dict:
        """Compare hyperparameter tuning studies."""
        # Implementation for hyperparameter comparison
        pass

def generate_comprehensive_report(model_name: str, 
                                include_training: bool = True,
                                include_testing: bool = True,
                                include_hyperparams: bool = False,
                                hyperparams_study_name: str = None) -> Dict:
    """
    Generate a comprehensive report for a model including all available analyses.
    
    Args:
        model_name: Name of the model to analyze
        include_training: Whether to include training analysis
        include_testing: Whether to include testing analysis  
        include_hyperparams: Whether to include hyperparameter analysis
        hyperparams_study_name: Name of hyperparameter study (if different from model_name)
        
    Returns:
        Dictionary containing all analysis results
    """
    
    generator = DQNReportsGenerator()
    results = {}
    
    try:
        if include_training:
            print("Generating training report...")
            results['training'] = generator.generate_training_report(model_name)
        
        if include_testing:
            print("Generating testing report...")
            results['testing'] = generator.generate_testing_report(model_name)
        
        if include_hyperparams:
            study_name = hyperparams_study_name or model_name
            print("Generating hyperparameter report...")
            
            # Extract components from model name for hyperparameter file lookup
            parts = model_name.split('_')
            if len(parts) >= 3:
                algorithm, reward_function, vsl_enforcement = parts[0], parts[1], parts[2]
                results['hyperparams'] = generator.generate_hyperparameter_tuning_report(
                    study_name, algorithm, reward_function, vsl_enforcement
                )
        
        print(f"Comprehensive report generated successfully for {model_name}")
        print(f"Reports saved to: {generator.output_dir}")
        
        return results
        
    except Exception as e:
        print(f"Error generating comprehensive report: {e}")
        return {}

if __name__ == "__main__":
    # Example usage
    
    # 1. Generate training report
    model_name = "DQN_balanced_recommend"
    generator = DQNReportsGenerator()
    
    try:
        # Training analysis
        # training_results = generator.generate_training_report(model_name)
        # print("Training report generated:", training_results)
        
        # # Testing analysis
        # testing_results = generator.generate_testing_report(model_name)
        # print("Testing report generated:", testing_results)
        
        # # Hyperparameter analysis (if available)
        # hp_results = generator.generate_hyperparameter_tuning_report(
        #     "DQN_study", "DQN", "balanced", "recommend"
        # )
        # print("Hyperparameter report generated:", hp_results)
        
        # Comprehensive report
        comprehensive_results = generate_comprehensive_report(
            model_name,
            include_training=True,
            include_testing=False,
            include_hyperparams=False
        )
        print("Comprehensive report generated:", comprehensive_results)

        # Check the TensorBoard logs for training metrics:
        # />tensorboard --logdir=./logs/[model trained]
        # NOTE: Check the rollout/ep_rew_mean and train/loss curves
        # when these stabilize for 50-100k timesteps, 
        # training has likely transitioned to a stable policy.

    except Exception as e:
        print(f"Report generation failed: {e}")
