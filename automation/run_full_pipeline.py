#!/usr/bin/env python3
"""
Full PI Testing Pipeline Automation
Runs: Test → Analysis → Plotting in one command

Usage:
    python run_full_pipeline.py --model gemini-2.0-flash --test my_test_name
    python run_full_pipeline.py --model gemini-2.0-flash --test my_test_name --skip-test  # Only analyze existing results
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime

def run_command(cmd, description, check=True):
    """Run a command and handle output"""
    print(f"\n🚀 {description}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=False, text=True)
        duration = time.time() - start_time
        print(f"\n✅ {description} completed in {duration:.1f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\n❌ {description} failed after {duration:.1f} seconds")
        print(f"Exit code: {e.returncode}")
        return False

def check_results_exist(test_name, model):
    """Check if test results already exist"""
    # Always look for eval_pi in the current directory (standalone repo structure)
    result_dir = f"eval_pi/{test_name}/{model}"
    
    if os.path.exists(result_dir):
        import glob
        test_dirs = glob.glob(os.path.join(result_dir, "*ntrk*"))
        return len(test_dirs), result_dir
    
    return 0, result_dir

def main():
    parser = argparse.ArgumentParser(description="Full PI Testing Pipeline")
    parser.add_argument("--model", required=True, help="Model name (e.g., gemini-2.0-flash)")
    parser.add_argument("--test", required=True, help="Test configuration name")
    parser.add_argument("--skip-test", action="store_true", help="Skip test execution, only analyze existing results")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis, only run tests")
    parser.add_argument("--skip-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--output-prefix", default="", help="Prefix for output files")
    
    args = parser.parse_args()
    
    print("🔬 PI Testing Full Pipeline")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Test: {args.test}")
    print(f"Skip test: {args.skip_test}")
    print(f"Skip analysis: {args.skip_analysis}")
    print(f"Skip plot: {args.skip_plot}")
    
    # Step 1: Run Tests (if not skipped)
    if not args.skip_test:
        print(f"\n📋 STEP 1: Running PI Tests")
        
        # Use relative path to core/run_pi.py in benchmark structure
        run_pi_path = "core/run_pi.py"
        
        cmd = ["python", run_pi_path, "--model", args.model, "--test", args.test]
        
        if not run_command(cmd, "PI Test Execution"):
            print("❌ Test execution failed. Stopping pipeline.")
            return 1
    else:
        print(f"\n⏭️  STEP 1: Skipped (test execution)")
    
    # Check if results exist
    num_results, result_dir = check_results_exist(args.test, args.model)
    if num_results == 0:
        print(f"❌ No test results found in {result_dir}")
        return 1
    
    print(f"✅ Found {num_results} test result directories in {result_dir}")
    
    # Step 2: Run Analysis (if not skipped)
    if not args.skip_analysis:
        print(f"\n📊 STEP 2: Running Analysis")
        
        # Use the same path as found by check_results_exist
        result_path = result_dir  # eval_pi in current directory
            
        cmd = ["python", "automation/analyze_pi_custom.py", result_path]
        
        if not run_command(cmd, "Result Analysis"):
            print("❌ Analysis failed. Continuing to plotting...")
        else:
            print("✅ Analysis completed successfully")
    else:
        print(f"\n⏭️  STEP 2: Skipped (analysis)")
    
    # Step 3: Create Plots (if not skipped)
    if not args.skip_plot:
        print(f"\n📈 STEP 3: Creating Accuracy Trend Plot")
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Use the same path logic as other functions
        # eval_pi in current directory (standalone repo structure)
        data_path = result_dir
        output_dir = f"eval_pi/plots_and_csv_beta/{args.test}_{args.model}"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        output_prefix = f"{output_dir}/{args.output_prefix}"
        
        # Update plot_accuracy_trend.py to use custom path
        plot_cmd = ["python", "-c", f"""
import sys
sys.path.append('automation')
from plot_accuracy_trend import collect_data, collect_data_len_item_mode, collect_data_nkeys_mode, create_plot, create_len_item_plots, create_nkeys_plots, extract_model_from_path, extract_test_params_from_path

# Extract model and test information
model_name = extract_model_from_path('{data_path}') or '{args.model}'
test_name, random_update = extract_test_params_from_path('{data_path}')
if not test_name:
    test_name = '{args.test}'

# Check test type
is_len_item_test = test_name and test_name.startswith('test_itemlen')
is_nkeys_test = test_name and test_name.startswith('test_nkeys')

if is_len_item_test:
    print('🔬 Detected len_item test - using len_item analysis mode')
    data = collect_data_len_item_mode('{data_path}')
    if data:
        output_file = '{output_prefix}accuracy_vs_len_item_{args.test}_{args.model.replace("-", "_")}_{timestamp}.png'
        result_dfs = create_len_item_plots(data, output_file, model_name, test_name)
        
        # Save CSV files for each n_tracked_updates
        for df in result_dfs:
            n_updates = df['n_tracked_updates'].iloc[0]
            csv_file = f'{output_prefix}accuracy_summary_{args.test}_{args.model.replace("-", "_")}_n_updates_{{n_updates}}_{timestamp}.csv'
            df.to_csv(csv_file, index=False)
            print(f'📄 Data saved as: {{csv_file}}')
    else:
        print('❌ No data to plot')
elif is_nkeys_test:
    print('🔑 Detected n_tracked_keys test - using n_tracked_keys analysis mode')
    data = collect_data_nkeys_mode('{data_path}')
    if data:
        output_file = '{output_prefix}accuracy_vs_nkeys_{args.test}_{args.model.replace("-", "_")}_{timestamp}.png'
        result_dfs = create_nkeys_plots(data, output_file, model_name, test_name)
        
        # Save CSV files for each n_tracked_updates
        for df in result_dfs:
            n_updates = df['n_tracked_updates'].iloc[0]
            csv_file = f'{output_prefix}accuracy_summary_{args.test}_{args.model.replace("-", "_")}_n_updates_{{n_updates}}_{timestamp}.csv'
            df.to_csv(csv_file, index=False)
            print(f'📄 Data saved as: {{csv_file}}')
    else:
        print('❌ No data to plot')
else:
    print('📈 Using default n_tracked_updates analysis mode')
    data = collect_data('{data_path}')
    if data:
        output_file = '{output_prefix}accuracy_vs_n_tracked_updates_{args.test}_{args.model.replace("-", "_")}_{timestamp}.png'
        df = create_plot(data, output_file, model_name, test_name, random_update)
        print(f'📊 Plot saved as: {{output_file}}')
        
        csv_file = '{output_prefix}accuracy_summary_{args.test}_{args.model.replace("-", "_")}_{timestamp}.csv'
        df.to_csv(csv_file, index=False)
        print(f'📄 Data saved as: {{csv_file}}')
    else:
        print('❌ No data to plot')
"""]
        
        if not run_command(plot_cmd, "Accuracy Trend Plotting"):
            print("❌ Plotting failed")
        else:
            print("✅ Plotting completed successfully")
    else:
        print(f"\n⏭️  STEP 3: Skipped (plotting)")
    
    # Final Summary
    print("\n" + "=" * 60)
    print("🎉 PIPELINE SUMMARY")
    print("=" * 60)
    print(f"✅ Model: {args.model}")
    print(f"✅ Test: {args.test}")
    print(f"✅ Results: {result_dir}")
    print(f"✅ Test directories: {num_results}")
    
    if not args.skip_plot:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"eval_pi/plots_and_csv_beta/{args.test}_{args.model}"
        plot_file = f"{output_dir}/accuracy_vs_n_tracked_updates_{args.test}_{args.model.replace('-', '_')}_{timestamp}.png"
        csv_file = f"{output_dir}/accuracy_summary_{args.test}_{args.model.replace('-', '_')}_{timestamp}.csv"
        print(f"📊 Plot: {plot_file}")
        print(f"📄 Data: {csv_file}")
    
    print("\n🎯 Pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 