# plot_results.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# --- Configuration ---
DATA_DIR = Path("paper_plots/data")
OUTPUT_DIR = Path("paper_plots/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define the models and their corresponding log files and nice names for plots
# IMPORTANT: Replace these with the actual filenames you generated
MODEL_FILES = {
    "No Control": "test_run_no_control_20250614_100000.csv",
    "DQN (Balanced)": "test_run_DQN_balanced_recommend_20250614_110000.csv",
    "DQN (Mobility)": "test_run_DQN_mobility_recommend_20250614_120000.csv",
    "DQN (Safety)": "test_run_DQN_safety_recommend_20250614_130000.csv",
}

# --- Plotting Style ---
# Use a professional style for the plots
sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

def load_and_process_data(model_files: dict) -> pd.DataFrame:
    """Loads all CSVs, adds a 'model' column, and concatenates them."""
    all_data = []
    for model_name, filename in model_files.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"Warning: File not found for '{model_name}': {filepath}")
            continue
        
        df = pd.read_csv(filepath)
        df['model'] = model_name
        
        # Convert simulation time to hours for better readability
        df['simulation_hours'] = df['simulation_time'] / 3600.0
        
        # Add a rolling average to smooth out noisy data for plotting
        df['flow_smoothed'] = df['flow_downstream'].rolling(window=10, min_periods=1).mean()
        df['queue_smoothed'] = df['queue_length'].rolling(window=10, min_periods=1).mean()
        
        all_data.append(df)
        
    if not all_data:
        raise FileNotFoundError("No valid data files were found. Check DATA_DIR and MODEL_FILES.")
        
    return pd.concat(all_data, ignore_index=True)

def plot_time_series(data: pd.DataFrame):
    """Plots key metrics over time for all models."""
    print("Generating time-series comparison plots...")
    
    # Plot 1: Smoothed Traffic Flow (Throughput)
    plt.figure()
    sns.lineplot(data=data, x='simulation_hours', y='flow_smoothed', hue='model', lw=2.5)
    plt.title('Traffic Flow Comparison Over Time')
    plt.xlabel('Simulation Time (hours)')
    plt.ylabel('Downstream Flow (vehicles/hour)')
    plt.legend(title='Control Strategy')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'timeseries_flow_comparison.png', dpi=300)
    plt.close()

    # Plot 2: Smoothed Queue Length
    plt.figure()
    sns.lineplot(data=data, x='simulation_hours', y='queue_smoothed', hue='model', lw=2.5)
    plt.title('Upstream Queue Length Comparison')
    plt.xlabel('Simulation Time (hours)')
    plt.ylabel('Queue Length (meters)')
    plt.legend(title='Control Strategy')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'timeseries_queue_comparison.png', dpi=300)
    plt.close()
    
    print("Time-series plots saved.")

def plot_summary_statistics(data: pd.DataFrame):
    """Creates bar charts comparing the overall performance of each model."""
    print("Generating summary statistics bar charts...")
    
    # Calculate summary metrics for each model
    summary = data.groupby('model').agg(
        avg_flow=('flow_downstream', 'mean'),
        max_queue=('queue_length', 'max'),
        avg_speed=('avg_speed_before_mps', lambda x: (x * 3.6).mean()), # Convert m/s to km/h
        speed_variance=('avg_speed_before_mps', 'var')
    ).reset_index()

    # Plot 1: Average Flow
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, x='model', y='avg_flow', palette='viridis')
    plt.title('Overall Performance: Average Traffic Flow')
    plt.xlabel('Control Strategy')
    plt.ylabel('Average Flow (vehicles/hour)')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_avg_flow.png', dpi=300)
    plt.close()

    # Plot 2: Maximum Queue Length
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, x='model', y='max_queue', palette='plasma')
    plt.title('Overall Performance: Maximum Queue Length')
    plt.xlabel('Control Strategy')
    plt.ylabel('Maximum Queue Length (meters)')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_max_queue.png', dpi=300)
    plt.close()
    
    print("Summary plots saved.")

def plot_tradeoff_analysis(data: pd.DataFrame):
    """Creates a scatter plot to show the trade-off between mobility and safety."""
    print("Generating trade-off analysis plot...")

    # For this plot, we'll look at the relationship between average flow and speed variance
    tradeoff_data = data.groupby('model').agg(
        avg_flow=('flow_downstream', 'mean'),
        speed_variance=('avg_speed_before_mps', 'var')
    ).reset_index()

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=tradeoff_data, x='speed_variance', y='avg_flow', hue='model', s=200, style='model', palette='deep')
    
    # Annotate points
    for i in range(tradeoff_data.shape[0]):
        plt.text(x=tradeoff_data.speed_variance[i]+0.1, y=tradeoff_data.avg_flow[i], s=tradeoff_data.model[i],
                 fontdict=dict(color='black', size=10))

    plt.title('Mobility vs. Safety Trade-off')
    plt.xlabel('Speed Variance (Lower is Safer)')
    plt.ylabel('Average Flow (Higher is More Mobile)')
    plt.legend(title='Control Strategy', loc='best')
    # Ideal corner is bottom-left (low variance, high flow)
    plt.axvline(x=tradeoff_data.speed_variance.min(), color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=tradeoff_data.avg_flow.max(), color='gray', linestyle='--', alpha=0.5)
    plt.annotate('Ideal Region', xy=(tradeoff_data.speed_variance.min(), tradeoff_data.avg_flow.max()), 
                 xytext=(tradeoff_data.speed_variance.min() + 1, tradeoff_data.avg_flow.max() - 200),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, backgroundcolor='w')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tradeoff_flow_vs_safety.png', dpi=300)
    plt.close()
    
    print("Trade-off plot saved.")


if __name__ == '__main__':
    try:
        # 1. Load and process data from all log files
        full_df = load_and_process_data(MODEL_FILES)
        
        # 2. Generate and save the time-series plots
        plot_time_series(full_df)
        
        # 3. Generate and save the summary bar charts
        plot_summary_statistics(full_df)
        
        # 4. Generate and save the trade-off scatter plot
        plot_tradeoff_analysis(full_df)
        
        print(f"\nAll plots have been successfully generated in the '{OUTPUT_DIR}' directory.")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure your DATA_DIR and MODEL_FILES variables in plot_results.py are correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")