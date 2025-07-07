# experiments/compare_sar.py
"""
Tools for comparing and analyzing different SAR configurations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class SARComparator:
    """Main class for comparing SAR configurations."""
    
    def __init__(self, results_dir: Path = None):
        # Set default to experiments/experiment_results
        if results_dir is None:
            results_dir = Path(__file__).parent / "experiment_results"
        self.results_dir = results_dir
        self.results = {}
        self.metrics_df = None
        
    def load_results(self, experiment_names: Optional[List[str]] = None):
        """Load experiment results from disk."""
        # Check if results directory exists
        if not self.results_dir.exists():
            logger.warning(f"Results directory does not exist: {self.results_dir}")
            logger.info("No experiments found. Please run some experiments first using:")
            logger.info("  python experiments/run_experiments.py --single --name test --state full_metrics --action absolute_speed --reward balanced")
            return
        
        if experiment_names is None:
            # Load all experiments
            experiment_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        else:
            experiment_dirs = []
            for name in experiment_names:
                exp_dir = self.results_dir / name
                if exp_dir.exists() and exp_dir.is_dir():
                    experiment_dirs.append(exp_dir)
                else:
                    logger.warning(f"Experiment directory not found: {exp_dir}")
        
        if not experiment_dirs:
            logger.warning(f"No experiment directories found in {self.results_dir}")
            logger.info("Available directories in results folder:")
            if self.results_dir.exists():
                for item in self.results_dir.iterdir():
                    logger.info(f"  - {item.name}")
            return
        
        for exp_dir in experiment_dirs:
            # Find the latest results file
            result_files = list(exp_dir.glob("*/results.json"))
            if not result_files:
                logger.warning(f"No results found in {exp_dir}")
                continue
            
            latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
            
            try:
                with open(latest_result, 'r') as f:
                    result_data = json.load(f)
                
                exp_name = exp_dir.name
                self.results[exp_name] = result_data
                logger.info(f"Loaded results for experiment: {exp_name}")
                
            except Exception as e:
                logger.error(f"Error loading results from {latest_result}: {e}")
                continue
            
        logger.info(f"Successfully loaded {len(self.results)} experiment results")
        
        # Build metrics dataframe only if we have results
        if self.results:
            self._build_metrics_dataframe()
        else:
            logger.warning("No results loaded. Cannot build metrics dataframe.")
    
    def _build_metrics_dataframe(self):
        """Build a pandas DataFrame from loaded results."""
        data = []
        
        for exp_name, result in self.results.items():
            if "error" in result:
                logger.warning(f"Skipping experiment {exp_name} due to error: {result['error']}")
                continue
                
            config = result["config"]
            metrics = result.get("metrics", {})
            
            row = {
                "experiment": exp_name,
                "state": config["state"],
                "action": config["action"],
                "reward": config["reward"],
                "algorithm": config.get("algorithm", "DQN"),
                "vsl_mode": config.get("vsl_mode", "recommend"),
            }
            
            # Add all metrics
            row.update(metrics)
            
            # Add evaluation details if available
            if "evaluation" in result:
                for scenario_name, scenario_results in result["evaluation"].items():
                    if "rewards" in scenario_results:
                        row[f"eval_{scenario_name}_reward"] = scenario_results["rewards"]["mean"]
                        row[f"eval_{scenario_name}_reward_std"] = scenario_results["rewards"]["std"]
            
            data.append(row)
        
        if data:
            self.metrics_df = pd.DataFrame(data)
            
            # Ensure numeric columns are float
            numeric_columns = [col for col in self.metrics_df.columns 
                              if col not in ["experiment", "state", "action", "reward", "algorithm", "vsl_mode"]]
            for col in numeric_columns:
                self.metrics_df[col] = pd.to_numeric(self.metrics_df[col], errors='coerce')
        else:
            logger.warning("No valid data to create metrics dataframe")
            self.metrics_df = pd.DataFrame()
    
    def rank_configurations(self, 
                          criteria: List[str] = None,
                          weights: Optional[List[float]] = None) -> pd.DataFrame:
        """Rank configurations based on specified criteria."""
        if self.metrics_df is None or self.metrics_df.empty:
            logger.warning("No metrics available for ranking")
            return pd.DataFrame()
        
        if criteria is None:
            criteria = ["avg_reward", "robustness_score"]
        
        if weights is None:
            weights = [1.0] * len(criteria)
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Check that all criteria exist
        missing = [c for c in criteria if c not in self.metrics_df.columns]
        if missing:
            logger.warning(f"Missing criteria columns: {missing}")
            available = [c for c in self.metrics_df.columns 
                        if c not in ["experiment", "state", "action", "reward", "algorithm", "vsl_mode"]]
            logger.info(f"Available criteria: {available}")
            criteria = [c for c in criteria if c in self.metrics_df.columns]
            weights = weights[:len(criteria)]
            
            if not criteria:
                logger.error("No valid criteria available for ranking")
                return pd.DataFrame()
        
        # Calculate composite score
        scores = pd.DataFrame()
        for criterion, weight in zip(criteria, weights):
            # Normalize to 0-1 range
            col_values = self.metrics_df[criterion].fillna(0)
            if col_values.max() > col_values.min():
                normalized = (col_values - col_values.min()) / (col_values.max() - col_values.min())
            else:
                normalized = pd.Series([0.5] * len(col_values))
            
            scores[criterion] = normalized * weight
        
        self.metrics_df["composite_score"] = scores.sum(axis=1)
        
        # Create ranking
        ranking = self.metrics_df.copy()
        ranking["rank"] = ranking["composite_score"].rank(ascending=False, method='min')
        
        # Sort by rank
        ranking = ranking.sort_values("rank")
        
        # Select relevant columns
        display_columns = ["rank", "experiment", "state", "action", "reward", 
                          "composite_score"] + criteria
        
        # Filter to existing columns
        display_columns = [col for col in display_columns if col in ranking.columns]
        
        return ranking[display_columns]
    
    def compare_by_component(self, component: str = "reward") -> pd.DataFrame:
        """Compare performance grouped by a specific SAR component."""
        if self.metrics_df is None or self.metrics_df.empty:
            logger.warning("No metrics available for comparison")
            return pd.DataFrame()
        
        if component not in ["state", "action", "reward"]:
            raise ValueError(f"Invalid component: {component}")
        
        # Find available metrics for aggregation
        metric_columns = ["avg_reward", "robustness_score"]
        available_metrics = [m for m in metric_columns if m in self.metrics_df.columns]
        
        if not available_metrics:
            logger.warning("No standard metrics available for comparison")
            return pd.DataFrame()
        
        # Build aggregation dict dynamically
        agg_dict = {}
        for metric in available_metrics:
            agg_dict[metric] = ["mean", "std", "min", "max"]
        
        # Group by component and calculate statistics
        grouped = self.metrics_df.groupby(component).agg(agg_dict)
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        
        # Add count
        grouped["count"] = self.metrics_df.groupby(component).size()
        
        return grouped
    
    def plot_performance_comparison(self, 
                                   metrics: List[str] = None,
                                   save_path: Optional[Path] = None):
        """Generate comparison plots for different metrics."""
        if self.metrics_df is None or self.metrics_df.empty:
            logger.warning("No data available for plotting")
            return
        
        if metrics is None:
            metrics = ["avg_reward", "robustness_score"]
        
        # Filter to available metrics
        metrics = [m for m in metrics if m in self.metrics_df.columns]
        
        if not metrics:
            logger.warning("No valid metrics to plot")
            available = [c for c in self.metrics_df.columns 
                        if c not in ["experiment", "state", "action", "reward", "algorithm", "vsl_mode"]]
            logger.info(f"Available metrics: {available}")
            return
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 3, figsize=(15, 5 * n_metrics))
        
        if n_metrics == 1:
            axes = axes.reshape(1, -1)
        
        components = ["state", "action", "reward"]
        
        for i, metric in enumerate(metrics):
            for j, component in enumerate(components):
                ax = axes[i, j]
                
                # Create box plot
                self.metrics_df.boxplot(column=metric, by=component, ax=ax)
                ax.set_title(f"{metric} by {component}")
                ax.set_xlabel(component)
                ax.set_ylabel(metric)
                
                # Rotate x labels if needed
                if component == "reward":
                    ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        else:
            plt.show()
    
    def plot_heatmap(self, 
                    metric: str = "avg_reward",
                    save_path: Optional[Path] = None):
        """Create a heatmap showing performance across SAR combinations."""
        if self.metrics_df is None or self.metrics_df.empty:
            logger.warning("No data available for heatmap")
            return
        
        if metric not in self.metrics_df.columns:
            logger.error(f"Metric '{metric}' not found in data")
            available = [c for c in self.metrics_df.columns 
                        if c not in ["experiment", "state", "action", "reward", "algorithm", "vsl_mode"]]
            logger.info(f"Available metrics: {available}")
            return
        
        # Pivot data for heatmap
        pivot_data = self.metrics_df.pivot_table(
            values=metric,
            index=['state', 'action'],
            columns='reward',
            aggfunc='mean'
        )
        
        if pivot_data.empty:
            logger.warning("Not enough data to create heatmap")
            return
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
        plt.title(f'{metric} Heatmap across SAR Configurations')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved heatmap to {save_path}")
        else:
            plt.show()
    
    def plot_scenario_performance(self, save_path: Optional[Path] = None):
        """Plot performance across different traffic scenarios."""
        if self.metrics_df is None or self.metrics_df.empty:
            logger.warning("No data available for scenario plot")
            return
        
        # Extract scenario columns
        scenario_cols = [col for col in self.metrics_df.columns 
                        if col.startswith("eval_") and col.endswith("_reward")]
        
        if not scenario_cols:
            logger.warning("No scenario evaluation data found")
            return
        
        # Prepare data for plotting
        scenario_data = []
        for _, row in self.metrics_df.iterrows():
            for col in scenario_cols:
                scenario = col.replace("eval_", "").replace("_reward", "")
                scenario_data.append({
                    "experiment": row["experiment"],
                    "sar_config": f"{row['state'][:3]}_{row['action'][:3]}_{row['reward'][:3]}",
                    "scenario": scenario,
                    "reward": row[col]
                })
        
        scenario_df = pd.DataFrame(scenario_data)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Use seaborn for better styling
        sns.boxplot(data=scenario_df, x="scenario", y="reward", hue="sar_config")
        plt.xticks(rotation=45)
        plt.title("Performance Across Traffic Scenarios")
        plt.ylabel("Average Reward")
        plt.xlabel("Traffic Scenario")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved scenario plot to {save_path}")
        else:
            plt.show()
    
    def generate_report(self, output_path: Path):
        """Generate a comprehensive comparison report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "n_experiments": len(self.results),
            "configurations": {},
            "rankings": {},
            "statistics": {},
            "recommendations": {}
        }
        
        if not self.results:
            report["error"] = "No experiment results found"
            logger.warning("No results to generate report")
            
            # Save empty report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            return report
        
        # Configuration summary
        for exp_name, result in self.results.items():
            if "error" not in result:
                config = result["config"]
                report["configurations"][exp_name] = {
                    "state": config["state"],
                    "action": config["action"],
                    "reward": config["reward"],
                    "metrics": result.get("metrics", {})
                }
        
        # Rankings (if we have data)
        if self.metrics_df is not None and not self.metrics_df.empty:
            ranking_df = self.rank_configurations()
            if not ranking_df.empty:
                report["rankings"]["overall"] = ranking_df.head(10).to_dict(orient='records')
                
                # Component-wise statistics
                for component in ["state", "action", "reward"]:
                    stats = self.compare_by_component(component)
                    if not stats.empty:
                        report["statistics"][component] = stats.to_dict()
                
                # Recommendations
                if len(ranking_df) > 0:
                    best_config = ranking_df.iloc[0]
                    report["recommendations"]["best_overall"] = {
                        "experiment": best_config["experiment"],
                        "state": best_config["state"],
                        "action": best_config["action"],
                        "reward": best_config["reward"],
                        "score": float(best_config.get("composite_score", 0))
                    }
            
            # Best for each objective
            objectives = {
                "mobility": ["flow_downstream_high_traffic", "avg_speed_before_high_traffic"],
                "safety": ["robustness_score"],
                "efficiency": ["avg_reward"]
            }
            
            for obj_name, criteria in objectives.items():
                valid_criteria = [c for c in criteria if c in self.metrics_df.columns]
                if valid_criteria:
                    obj_ranking = self.rank_configurations(criteria=valid_criteria)
                    if len(obj_ranking) > 0:
                        best = obj_ranking.iloc[0]
                        report["recommendations"][f"best_for_{obj_name}"] = {
                            "experiment": best["experiment"],
                            "state": best["state"],
                            "action": best["action"],
                            "reward": best["reward"],
                        }
        else:
            report["note"] = "Insufficient data for rankings and statistics"
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated comparison report: {output_path}")
        
        return report
    
    def export_to_csv(self, output_path: Path):
        """Export all metrics to CSV for further analysis."""
        if self.metrics_df is None or self.metrics_df.empty:
            logger.warning("No data to export")
            return
        
        self.metrics_df.to_csv(output_path, index=False)
        logger.info(f"Exported metrics to {output_path}")


# Convenience functions
def load_experiment_results(experiment_names: Optional[List[str]] = None,
                           results_dir: Path = None) -> SARComparator:
    """Load experiment results and return comparator object."""
    comparator = SARComparator(results_dir)
    comparator.load_results(experiment_names)
    return comparator


def generate_comparison_report(experiment_names: Optional[List[str]] = None,
                             output_path: Optional[Path] = None,
                             results_dir: Path = None) -> Dict[str, Any]:
    """Generate a comparison report for specified experiments."""
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(f"comparison_report_{timestamp}.json")
    
    comparator = load_experiment_results(experiment_names, results_dir)
    return comparator.generate_report(output_path)


def plot_performance_comparison(experiment_names: Optional[List[str]] = None,
                               metrics: List[str] = None,
                               save_path: Optional[Path] = None,
                               results_dir: Path = None):
    """Plot performance comparison for specified experiments."""
    comparator = load_experiment_results(experiment_names, results_dir)
    comparator.plot_performance_comparison(metrics, save_path)


def rank_configurations(experiment_names: Optional[List[str]] = None,
                       criteria: List[str] = None,
                       weights: Optional[List[float]] = None,
                       results_dir: Path = None) -> pd.DataFrame:
    """Rank configurations based on specified criteria."""
    comparator = load_experiment_results(experiment_names, results_dir)
    return comparator.rank_configurations(criteria, weights)


def main():
    """Main entry point for comparison tools."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare SAR configurations")
    
    parser.add_argument('--action', type=str, required=True,
                       choices=['report', 'rank', 'plot', 'export'],
                       help='Action to perform')
    
    parser.add_argument('--experiments', type=str, nargs='+',
                       help='Specific experiments to compare (default: all)')
    
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Directory containing experiment results (default: experiments/experiment_results)')
    
    parser.add_argument('--output', type=str,
                       help='Output file path')
    
    # Ranking options
    parser.add_argument('--criteria', type=str, nargs='+',
                       default=['avg_reward', 'robustness_score'],
                       help='Criteria for ranking')
    
    parser.add_argument('--weights', type=float, nargs='+',
                       help='Weights for ranking criteria')
    
    # Plotting options
    parser.add_argument('--metrics', type=str, nargs='+',
                       default=['avg_reward', 'robustness_score'],
                       help='Metrics to plot')
    
    parser.add_argument('--plot-type', type=str, default='comparison',
                       choices=['comparison', 'heatmap', 'scenario'],
                       help='Type of plot to generate')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Default to experiments/experiment_results
        results_dir = Path(__file__).parent / "experiment_results"
    
    # Check if results directory exists
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        logger.info("\nTo fix this issue:")
        logger.info("1. Run some experiments first:")
        logger.info("   python experiments/run_experiments.py --single --name test --state full_metrics --action absolute_speed --reward balanced")
        logger.info("\n2. Or create the directory manually:")
        logger.info(f"   mkdir {results_dir}")
        logger.info("\n3. Or specify a different results directory:")
        logger.info("   python experiments/compare_sar.py --action report --results-dir path/to/results")
        return
    
    if args.action == 'report':
        output_path = Path(args.output) if args.output else None
        report = generate_comparison_report(args.experiments, output_path, results_dir)
        
        if "error" in report:
            logger.error(f"Report generation failed: {report['error']}")
        else:
            logger.info(f"Generated report with {len(report['configurations'])} experiments")
            if output_path:
                logger.info(f"Report saved to: {output_path}")
        
    elif args.action == 'rank':
        ranking = rank_configurations(
            args.experiments,
            args.criteria,
            args.weights,
            results_dir
        )
        
        if ranking.empty:
            logger.error("No data available for ranking")
        else:
            print("\nTop 10 Configurations:")
            print(ranking.head(10).to_string())
            
            if args.output:
                ranking.to_csv(args.output, index=False)
                print(f"\nFull ranking saved to {args.output}")
    
    elif args.action == 'plot':
        comparator = load_experiment_results(args.experiments, results_dir)
        
        if not comparator.results:
            logger.error("No results loaded for plotting")
            return
        
        save_path = Path(args.output) if args.output else None
        
        if args.plot_type == 'comparison':
            comparator.plot_performance_comparison(args.metrics, save_path)
        elif args.plot_type == 'heatmap':
            comparator.plot_heatmap(args.metrics[0], save_path)
        elif args.plot_type == 'scenario':
            comparator.plot_scenario_performance(save_path)
    
    elif args.action == 'export':
        if not args.output:
            args.output = f"experiment_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        comparator = load_experiment_results(args.experiments, results_dir)
        
        if comparator.results:
            comparator.export_to_csv(Path(args.output))
            print(f"Exported {len(comparator.metrics_df)} experiments to {args.output}")
        else:
            logger.error("No results to export")


if __name__ == "__main__":
    main()