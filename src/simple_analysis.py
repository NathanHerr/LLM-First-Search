#!/usr/bin/env python3
"""
Simple script to analyze and compare different methods across tasks.
Automatically detects available methods and tasks from data directories.

Features:
- Automatic detection of methods and tasks from pickle files
- Win rate comparison plots
- Token usage analysis
- Efficiency analysis (win rate vs token usage)
- AUP (Area Under Performance Profile) calculation for both win rate and efficiency metrics
- Excel and CSV export with multiple sheets/files

AUP Metrics:
- Win Rate AUP: Measures how consistently a method performs well across tasks
- Efficiency AUP: Combines win rate and token efficiency for overall performance assessment

Usage:
    python simple_analysis.py --data_dir /path/to/results --output_dir /path/to/output

This will automatically:
- Detect all available methods and tasks
- Generate win rate comparisons
- Create performance summaries
- Calculate AUP (Area Under Performance Profile) values
- Save results as plots and Excel files
"""

import argparse
import glob
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Try to import openpyxl for Excel support, fall back to xlsxwriter
try:
    import openpyxl
    EXCEL_ENGINE = 'openpyxl'
except ImportError:
    try:
        import xlsxwriter
        EXCEL_ENGINE = 'xlsxwriter'
    except ImportError:
        EXCEL_ENGINE = None
        print("⚠️  Warning: Neither openpyxl nor xlsxwriter found. Excel output may not work.")

# Add the project root to Python path for pickle loading
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Also add the src directory to ensure relative imports work
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Now try to import the required class
try:
    from llm_mcts import MonteCarloTreeSearchNode
    MCTS_AVAILABLE = True
    print("✅ Successfully imported MonteCarloTreeSearchNode")
except ImportError as e:
    MCTS_AVAILABLE = False
    print(f"⚠️  Warning: Could not import MonteCarloTreeSearchNode: {e}")
    print("   Some MCTS pickle files may not load properly")

def safe_pickle_load(file_path):
    """
    Safely load pickle files with missing dependencies.
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except ImportError as e:
        print(f"      Warning: Import error loading {os.path.basename(file_path)}: {e}")
        return None
    except AttributeError as e:
        print(f"      Warning: Attribute error loading {os.path.basename(file_path)}: {e}")
        return None
    except Exception as e:
        print(f"      Warning: Failed to load {os.path.basename(file_path)}: {e}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple analysis of method results across tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis - auto-detect everything
    python simple_analysis.py --data_dir ./results --output_dir ./analysis
    
    # Specify model
    python simple_analysis.py --data_dir ./results --output_dir ./analysis --model gpt-4o
    
    # Include only specific methods
    python simple_analysis.py --data_dir ./results --output_dir ./analysis --methods "mcts,bestfs,lfs"
        """
    )
    
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing result pickle files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save analysis results")
    parser.add_argument("--model", type=str, default="gpt-4o",
                       help="Model name to analyze (default: gpt-4o)")
    parser.add_argument("--methods", type=str, default=None,
                       help="Comma-separated methods to include (default: auto-detect all)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size used in experiments (default: 1)")
    
    return parser.parse_args()

def auto_detect_available_data(data_dir, model_name, batch_size):
    """
    Automatically detect available methods and tasks from the data directory.
    
    Returns:
        dict: {'methods': set, 'tasks': set, 'files': list}
    """
    print(f"🔍 Auto-detecting available data in {data_dir}...")
    
    # Find all pickle files
    pickle_files = glob.glob(os.path.join(data_dir, "**", "*.pkl"), recursive=True)
    
    detected_methods = set()
    detected_tasks = set()
    valid_files = []
    
    for file_path in pickle_files:
        # Extract info from filename
        filename = os.path.basename(file_path)
        
        # Skip if wrong model
        if model_name not in filename:
            continue
            
        # Extract method from filename - methods appear before _mtu_
        method_patterns = {
            'mcts': r'mcts',  # mcts but not llm_mcts (negative lookbehind)
            'bestfs': r'bestfs',
            'tot_bfs': r'tot_bfs',
            'lfs': r'lfs',
        }
        
        method_found = None
        for method, pattern in method_patterns.items():
            if re.search(pattern, filename):
                method_found = method
                break
        
        if not method_found:
            continue
            
        # Determine task type and parameters
        task_name = None
        
        # Countdown pattern: batch_X_size_X_val_X_responses_model_temp_method_mtu_X.pkl
        countdown_match = re.search(r'batch_\d+_size_\d+_val_(\d+)_responses_', filename)
        if countdown_match:
            difficulty = countdown_match.group(1)
            task_name = f"Countdown (diff={difficulty})"
        
        # Sudoku pattern: batch_X_size_X_sudoku_w_X_h_X_difficulty_responses_model_temp_method_mtu_X.pkl
        sudoku_match = re.search(r'batch_\d+_size_\d+_sudoku_w_(\d+)_h_(\d+)_(\w+)_responses_', filename)
        if sudoku_match:
            width = sudoku_match.group(1)
            height = sudoku_match.group(2)
            difficulty = sudoku_match.group(3)
            task_name = f"Sudoku ({width}x{height}, {difficulty})"
        
        if task_name:
            detected_methods.add(method_found)
            detected_tasks.add(task_name)
            valid_files.append({
                'path': file_path,
                'method': method_found,
                'task': task_name
            })
    
    print(f"✅ Found {len(detected_methods)} methods: {sorted(detected_methods)}")
    print(f"✅ Found {len(detected_tasks)} tasks: {sorted(detected_tasks)}")
    print(f"✅ Found {len(valid_files)} valid result files")
    
    return {
        'methods': detected_methods,
        'tasks': detected_tasks, 
        'files': valid_files
    }

def load_and_analyze_data(detected_data, methods_filter=None):
    """
    Load data and extract key metrics.
    
    Returns:
        dict: {task_name: {method: metrics}}
    """
    print("📊 Loading and analyzing data...")
    
    # Filter methods if specified
    if methods_filter:
        methods_to_include = set(methods_filter.split(','))
        files_to_process = [f for f in detected_data['files'] 
                          if f['method'] in methods_to_include]
    else:
        files_to_process = detected_data['files']
    
    # Group files by task and method
    task_method_files = defaultdict(lambda: defaultdict(list))
    for file_info in files_to_process:
        task_method_files[file_info['task']][file_info['method']].append(file_info['path'])
    
    # Load data and compute metrics
    results = {}
    
    for task_name, method_files in task_method_files.items():
        print(f"  Processing {task_name}...")
        results[task_name] = {}
        
        for method_name, file_paths in method_files.items():
            print(f"    Loading {method_name}...")
            
            # Load all data for this method+task combination
            all_games = []
            for file_path in file_paths:
                batch_data = safe_pickle_load(file_path)
                if batch_data is not None:
                    all_games.extend(batch_data)
            
            if all_games:
                metrics = compute_metrics(all_games, method_name, task_name)
                results[task_name][method_name] = metrics
    
    return results

def compute_metrics(games_data, method_name, task_name):
    """
    Compute key metrics from games data.
    
    Returns:
        dict: Key performance metrics
    """
    if not games_data:
        return None
    
    total_games = len(games_data)
    game_win_rates = []
    game_token_usages = []
    total_attempts = 0
    total_wins = 0
    
    for game in games_data:
        # Count wins and attempts for this game
        game_attempts = len(game)
        game_wins = sum(1 for attempt in game if attempt.get("won", False))
        
        total_attempts += game_attempts
        total_wins += game_wins
        
        # Game-level win rate
        game_win_rate = game_wins / game_attempts if game_attempts > 0 else 0
        game_win_rates.append(game_win_rate)
        
        # Average tokens for this game
        game_tokens = []
        for attempt in game:
            if "total_tokens" in attempt:
                game_tokens.append(attempt["total_tokens"])
            elif "token_usage" in attempt:
                # Handle nested token usage structure
                tokens = 0
                for _, usage in attempt["token_usage"]:
                    if usage and "total_tokens" in usage:
                        tokens += usage["total_tokens"]
                game_tokens.append(tokens)
        
        if game_tokens:
            game_token_usages.append(np.mean(game_tokens))
    
    # Overall metrics
    overall_win_rate = np.mean(game_win_rates) if game_win_rates else 0
    avg_tokens = np.mean(game_token_usages) if game_token_usages else 0
    
    print(f"      {method_name}: {overall_win_rate:.1%} win rate, {avg_tokens:.0f} avg tokens")
    
    return {
        'method': method_name,
        'task': task_name,
        'total_games': total_games,
        'total_attempts': total_attempts,
        'total_wins': total_wins,
        'overall_win_rate': overall_win_rate,
        'avg_tokens': avg_tokens,
        'game_win_rates': game_win_rates,
        'game_token_usages': game_token_usages
    }

def calculate_aup_values(results):
    """
    Calculate AUP (Area Under Performance Profile) values for WinRate and EfficiencyScore.
    
    Args:
        results: Dictionary of {task_name: {method: metrics}}
    
    Returns:
        dict: AUP values for each method and metric type
    """
    print("📊 Calculating AUP values...")
    
    # Prepare tau values for performance profile (logarithmic scale from 1 to 10)
    tau_values = np.logspace(0, np.log10(10), 200)
    
    # Initialize AUP results
    aup_results = {
        'WinRate': {},
        'EfficiencyScore': {}
    }
    
    # Get all methods
    all_methods = set()
    for task_results in results.values():
        all_methods.update(task_results.keys())
    
    # Calculate performance ratios for each metric
    method_ratios = {
        'WinRate': {},
        'EfficiencyScore': {}
    }
    
    # First pass: find best performance for each task
    best_tokens = {}
    best_win_rates = {}
    
    for task_name, task_results in results.items():
        best_tokens[task_name] = float('inf')
        best_win_rates[task_name] = float('-inf')
        
        # Find best values for each metric
        for method_name, metrics in task_results.items():
            if metrics is None:
                continue
            
            best_tokens[task_name] = min(best_tokens[task_name], metrics["avg_tokens"])
            best_win_rates[task_name] = max(best_win_rates[task_name], metrics["overall_win_rate"])
    
    # Second pass: calculate ratios and AUP for each method
    for method_name in all_methods:
        win_rate_ratios = []
        efficiency_score_ratios = []
        
        for task_name, task_results in results.items():
            metrics = task_results.get(method_name)
            
            if metrics is None:
                # If method doesn't have data for this task, assign infinite ratio (worst performance)
                win_rate_ratios.append(float('inf'))
                efficiency_score_ratios.append(float('inf'))
                continue
            
            # Win rate ratio: best_win_rate / method_win_rate
            if metrics["overall_win_rate"] > 0:
                win_rate_ratio = best_win_rates[task_name] / metrics["overall_win_rate"]
            else:
                win_rate_ratio = float('inf')
            win_rate_ratios.append(win_rate_ratio)
            
            # Efficiency score ratio: (best_win_rate/win_rate) * (tokens/best_tokens)
            if metrics["overall_win_rate"] > 0 and best_tokens[task_name] > 0:
                wr_ratio = best_win_rates[task_name] / metrics["overall_win_rate"]
                token_ratio = metrics["avg_tokens"] / best_tokens[task_name]
                efficiency_score_ratio = wr_ratio * token_ratio
            else:
                efficiency_score_ratio = float('inf')
            efficiency_score_ratios.append(efficiency_score_ratio)
        
        # Store ratios
        method_ratios['WinRate'][method_name] = win_rate_ratios
        method_ratios['EfficiencyScore'][method_name] = efficiency_score_ratios
        
        # Calculate AUP for each metric
        for metric_name in ['WinRate', 'EfficiencyScore']:
            ratios = method_ratios[metric_name][method_name]
            
            # Calculate performance profile
            profile = []
            for tau in tau_values:
                count = sum(1 for r in ratios if r <= tau)
                profile.append(count / len(ratios) if len(ratios) > 0 else 0)
            
            # Calculate AUP using trapezoidal integration
            aup = np.trapz(profile, tau_values)
            aup_results[metric_name][method_name] = aup
            
            print(f"    {method_name} - {metric_name}: AUP = {aup:.3f}")
    
    return aup_results

def create_summary_plots(results, output_dir, model_name):
    """
    Create simple, clear summary plots including AUP values.
    """
    print("📈 Creating summary plots...")
    
    # Set style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10
    })
    
    # Prepare data
    plot_data = []
    for task_name, task_results in results.items():
        for method_name, metrics in task_results.items():
            if metrics:
                plot_data.append({
                    'Task': task_name,
                    'Method': method_name,
                    'Win Rate (%)': metrics['overall_win_rate'] * 100,
                    'Avg Tokens': metrics['avg_tokens']
                })
    
    df = pd.DataFrame(plot_data)
    
    if df.empty:
        print("❌ No data to plot")
        return
    
    # 1. Win Rate Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Task', y='Win Rate (%)', hue='Method')
    plt.title(f'Win Rate Comparison - {model_name}')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'win_rates.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Token Usage Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Task', y='Avg Tokens', hue='Method')
    plt.title(f'Average Token Usage - {model_name}')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_usage.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Efficiency Plot (Win Rate vs Tokens)
    plt.figure(figsize=(10, 6))
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        plt.scatter(method_data['Avg Tokens'], method_data['Win Rate (%)'], 
                   label=method, s=100, alpha=0.7)
    
    plt.xlabel('Average Token Usage')
    plt.ylabel('Win Rate (%)')
    plt.title(f'Efficiency: Win Rate vs Token Usage - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. AUP Values Bar Plot
    aup_values = calculate_aup_values(results)
    
    # Prepare AUP data for plotting
    aup_plot_data = []
    for metric_type in ['WinRate', 'EfficiencyScore']:
        for method, aup in aup_values[metric_type].items():
            aup_plot_data.append({
                'Method': method,
                'Metric': 'Win Rate AUP' if metric_type == 'WinRate' else 'Efficiency Score AUP',
                'AUP': aup
            })
    
    aup_df = pd.DataFrame(aup_plot_data)
    
    if not aup_df.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=aup_df, x='Method', y='AUP', hue='Metric')
        plt.title(f'AUP (Area Under Performance Profile) - {model_name}')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aup_values.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Plots saved to {output_dir}")

def create_summary_table(results, output_dir, model_name):
    """
    Create Excel summary table including AUP values.
    """
    print("📋 Creating summary table...")
    
    # Calculate AUP values
    aup_values = calculate_aup_values(results)
    
    # Prepare data
    summary_data = []
    for task_name, task_results in results.items():
        for method_name, metrics in task_results.items():
            if metrics:
                summary_data.append({
                    'Task': task_name,
                    'Method': method_name,
                    'Total Games': metrics['total_games'],
                    'Win Rate (%)': f"{metrics['overall_win_rate']*100:.1f}",
                    'Avg Tokens': f"{metrics['avg_tokens']:.0f}",
                    'Total Attempts': metrics['total_attempts'],
                    'Total Wins': metrics['total_wins']
                })
    
    # Prepare AUP summary data
    aup_summary_data = []
    for method_name in aup_values['WinRate'].keys():
        aup_summary_data.append({
            'Method': method_name,
            'Win Rate AUP': f"{aup_values['WinRate'][method_name]:.3f}",
            'Efficiency Score AUP': f"{aup_values['EfficiencyScore'][method_name]:.3f}"
        })
    
    df = pd.DataFrame(summary_data)
    aup_df = pd.DataFrame(aup_summary_data)
    
    if not df.empty:
        # Save main results as Excel with multiple sheets
        excel_path = os.path.join(output_dir, 'summary.xlsx')
        if EXCEL_ENGINE:
            with pd.ExcelWriter(excel_path, engine=EXCEL_ENGINE) as writer:
                df.to_excel(writer, sheet_name='Results', index=False)
                if not aup_df.empty:
                    aup_df.to_excel(writer, sheet_name='AUP Values', index=False)
            print(f"✅ Summary Excel file saved to {excel_path}")
        else:
            print("⚠️  Skipping Excel file creation (no Excel engine available)")
        
        # Save as CSV too
        csv_path = os.path.join(output_dir, 'summary.csv')
        df.to_csv(csv_path, index=False)
        
        aup_csv_path = os.path.join(output_dir, 'aup_summary.csv')
        if not aup_df.empty:
            aup_df.to_csv(aup_csv_path, index=False)
        
        print(f"✅ Summary tables saved to {csv_path}")
        if not aup_df.empty:
            print(f"✅ AUP summary saved to {aup_csv_path}")
        
        # Print summary to console
        print("\n📊 SUMMARY:")
        print("=" * 80)
        for task in df['Task'].unique():
            task_data = df[df['Task'] == task]
            print(f"\n{task}:")
            print("-" * 40)
            for _, row in task_data.iterrows():
                print(f"  {row['Method']:12s}: {row['Win Rate (%)']}% win rate, {row['Avg Tokens']} tokens")
        
        # Print AUP summary
        if not aup_df.empty:
            print("\n📊 AUP VALUES:")
            print("=" * 80)
            for _, row in aup_df.iterrows():
                print(f"  {row['Method']:12s}: Win Rate AUP = {row['Win Rate AUP']}, Efficiency Score AUP = {row['Efficiency Score AUP']}")
    else:
        print("❌ No data for summary table")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 Starting simple analysis for model: {args.model}")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Auto-detect available data
    detected_data = auto_detect_available_data(args.data_dir, args.model, args.batch_size)
    
    if not detected_data['files']:
        print("❌ No valid data files found!")
        print("Make sure:")
        print("  - Data directory contains .pkl files")
        print(f"  - Files contain model name '{args.model}'")
        print("  - Files follow expected naming convention")
        return
    
    # Load and analyze data
    results = load_and_analyze_data(detected_data, args.methods)
    
    if not results:
        print("❌ No results generated!")
        return
    
    # Create visualizations
    create_summary_plots(results, args.output_dir, args.model)
    
    # Create summary table
    create_summary_table(results, args.output_dir, args.model)
    
    print(f"\n🎉 Analysis complete! Results saved to {args.output_dir}")
    print("\nGenerated files:")
    print("  - win_rates.png: Win rate comparison across tasks")
    print("  - token_usage.png: Average token usage comparison") 
    print("  - efficiency.png: Win rate vs token usage scatter plot")
    print("  - aup_values.png: AUP (Area Under Performance Profile) comparison")
    print("  - summary.xlsx: Detailed results table with AUP values")
    print("  - summary.csv: Detailed results table (CSV format)")
    print("  - aup_summary.csv: AUP values summary (CSV format)")
    print("\nAUP Metrics:")
    print("  - Win Rate AUP: Consistency of win rate across tasks")
    print("  - Efficiency Score AUP: Combined win rate and token efficiency")

if __name__ == "__main__":
    main() 