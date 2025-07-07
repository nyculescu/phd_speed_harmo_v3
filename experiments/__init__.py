# experiments/__init__.py
"""
Experiments Module for DRL-VSL Framework

This module provides tools for running systematic experiments with different
SAR configurations and comparing their performance.
"""

from .run_experiments import (
    ExperimentRunner,
    ExperimentConfig,
    run_single_experiment,
    run_batch_experiments,
    run_from_config_file,
)

from .compare_sar import (
    SARComparator,
    load_experiment_results,
    generate_comparison_report,
    plot_performance_comparison,
    rank_configurations,
)

__all__ = [
    # Runner
    "ExperimentRunner",
    "ExperimentConfig",
    "run_single_experiment",
    "run_batch_experiments", 
    "run_from_config_file",
    
    # Comparator
    "SARComparator",
    "load_experiment_results",
    "generate_comparison_report",
    "plot_performance_comparison",
    "rank_configurations",
]