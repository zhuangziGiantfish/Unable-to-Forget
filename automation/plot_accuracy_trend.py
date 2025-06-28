#!/usr/bin/env python3
"""
Plot accuracy vs n_tracked_updates to visualize the trend
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from config_utils import get_n_tracked_keys_for_test, create_directory_pattern
import scipy.stats as stats

def compute_ci95_width_by_bootstrap(acc, n=5000):
    """Compute the width of the 95% confidence interval by bootstrapping."""
    if len(acc) <= 1:
        return np.nan, 0

    acc = np.array(acc)
    # 进行bootstrap采样
    arr = [np.mean(np.random.choice(acc, len(acc), True)) for _ in range(n)]
    arr.sort()
    width = np.percentile(arr, 97.5) - np.percentile(arr, 2.5)
    return width, len(acc)

def extract_n_tracked_updates(dir_name):
    """Extract n_tracked_updates value from directory name"""
    if "_ntrkupd-" in dir_name:
        try:
            return int(dir_name.split("_ntrkupd-")[1].split("_")[0])
        except:
            return None
    return None

def extract_len_item(dir_name):
    """Extract len_item value from directory name"""
    if "_lenitm-" in dir_name:
        try:
            return int(dir_name.split("_lenitm-")[1].split("_")[0])
        except:
            return None
    return None

def extract_accuracy_from_csv(csv_file):
    """Extract accuracy from CSV file and calculate mean and 95% CI using bootstrap method"""
    try:
        df = pd.read_csv(csv_file)
        if 'accuracy' in df.columns and not df.empty:
            # 统计总数据和有效数据
            total_rows = len(df)
            accuracies_with_nan = df['accuracy']
            accuracies = accuracies_with_nan.dropna().values
            valid_rows = len(accuracies)
            nan_rows = total_rows - valid_rows
            
            if len(accuracies) == 0:
                print(f"Warning: No valid accuracy data in {csv_file}")
                return None
            
            # 如果有缺失数据，报告一下
            if nan_rows > 0:
                print(f"Info: {csv_file} - {nan_rows}/{total_rows} rows with missing accuracy data were excluded")
            
            mean_acc = np.mean(accuracies)
            ci95_width, n = compute_ci95_width_by_bootstrap(accuracies)
            return {
                'mean': mean_acc,
                'ci95': ci95_width / 2,  # 转换为单边误差
                'n': n,  # 有效数据数量
                'total_rows': total_rows,  # 总行数
                'valid_rows': valid_rows,  # 有效行数
                'excluded_rows': nan_rows,  # 被排除的行数
                'raw_data': df  # 添加原始数据
            }
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
    return None

def collect_data(base_dir):
    """Collect n_tracked_updates and accuracy data from all test directories"""
    data = []
    raw_data_list = []  # 用于存储所有原始数据
    
    # Try to determine test name from path for config lookup
    test_name = None
    path_parts = base_dir.replace('\\', '/').split('/')
    for part in path_parts:
        if part.startswith('test_'):
            test_name = part
            break
    
    # Try to get n_tracked_keys from config
    n_tracked_keys = None
    if test_name:
        n_tracked_keys = get_n_tracked_keys_for_test(test_name)
        if n_tracked_keys:
            print(f"📋 Found n_tracked_keys={n_tracked_keys} from config for test '{test_name}'")
    
    # If we couldn't get it from config, try to infer from existing directories
    if n_tracked_keys is None:
        print("⚠️  Could not get n_tracked_keys from config, scanning for existing patterns...")
        # Look for any directory with ntrk pattern
        all_dirs = glob.glob(os.path.join(base_dir, "*ntrk-*_ntrkupd*"))
        if all_dirs:
            # Extract n_tracked_keys from the first directory found
            for dir_path in all_dirs:
                dir_name = os.path.basename(dir_path)
                if "_ntrk-" in dir_name and "_ntrkupd-" in dir_name:
                    try:
                        n_tracked_keys = int(dir_name.split("_ntrk-")[1].split("_")[0])
                        print(f"📋 Inferred n_tracked_keys={n_tracked_keys} from existing directory pattern")
                        break
                    except:
                        continue
    
    # Create pattern based on n_tracked_keys
    if n_tracked_keys is not None:
        pattern = os.path.join(base_dir, create_directory_pattern(n_tracked_keys))
        print(f"🔍 Using pattern: {create_directory_pattern(n_tracked_keys)}")
    else:
        # Fallback to old hardcoded pattern (for backward compatibility)
        pattern = os.path.join(base_dir, "*ntrk-46_ntrkupd*")
        print(f"⚠️  Falling back to legacy pattern: *ntrk-46_ntrkupd*")
    
    # Find all test directories
    test_dirs = sorted(glob.glob(pattern))
    
    print(f"Found {len(test_dirs)} test directories")
    
    for test_dir in test_dirs:
        dir_name = os.path.basename(test_dir)
        n_updates = extract_n_tracked_updates(dir_name)
        
        if n_updates is None:
            continue
            
        # Find the CSV file
        csv_pattern = os.path.join(test_dir, "*_trial.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f"Warning: No CSV file found in {dir_name}")
            continue
            
        csv_file = csv_files[0]  # Take the first one
        accuracy = extract_accuracy_from_csv(csv_file)
        
        if accuracy is not None:
            data.append({
                'n_tracked_updates': n_updates,
                'accuracy_mean': accuracy['mean'],
                'accuracy_ci95': accuracy['ci95'],
                'n': accuracy['n'],
                'directory': dir_name
            })
            
            # 添加原始数据到列表
            if 'raw_data' in accuracy:
                raw_df = accuracy['raw_data']
                raw_df['n_tracked_updates'] = n_updates
                raw_df['directory'] = dir_name
                raw_data_list.append(raw_df)
            
            print(f"✅ n_tracked_updates={n_updates:3d}, accuracy={accuracy['mean']:.3f} ± {accuracy['ci95']:.3f}")
        else:
            print(f"❌ Failed to extract accuracy from {dir_name}")
    
    # 保存原始数据到CSV文件
    if raw_data_list:
        raw_data_df = pd.concat(raw_data_list, ignore_index=True)
        # 获取当前工作目录
        current_dir = os.getcwd()
        # 生成时间戳
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        # 从base_dir中提取模型名称
        model_name = os.path.basename(base_dir)
        # 构建文件名
        raw_data_file = os.path.join(current_dir, f'raw_accuracy_data_{test_name}_{model_name}_{timestamp}.csv')
        raw_data_df.to_csv(raw_data_file, index=False)
        print(f"\n📄 Raw data saved as: {raw_data_file}")
    
    return data

def create_plot(data, output_file="accuracy_vs_n_tracked_updates.png", model_name=None, test_name=None, random_update=None):
    """Create a plot showing accuracy vs n_tracked_updates"""
    if not data:
        print("No data to plot!")
        return
        
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(data)
    df = df.sort_values('n_tracked_updates')
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot with error bars
    plt.errorbar(
        df['n_tracked_updates'],
        df['accuracy_mean'],
        yerr=df['accuracy_ci95'],
        fmt='o-',
        capsize=5,
        markersize=8,
        label='Mean with 95% CI'
    )
    
    # Add some statistics
    max_acc = df['accuracy_mean'].max()
    min_acc = df['accuracy_mean'].min()
    mean_acc = df['accuracy_mean'].mean()
    
    # Add text box with statistics
    stats_text = f'Max Accuracy: {max_acc:.3f}\nMin Accuracy: {min_acc:.3f}\nMean Accuracy: {mean_acc:.3f}\nData Points: {len(df)}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    # Use log scale for better visualization of the wide range
    plt.xscale('log')
    
    # Set custom x-axis ticks to show each data point exactly
    plt.xticks(df['n_tracked_updates'], labels=df['n_tracked_updates'].astype(str), rotation=45, ha='right')
    plt.xlabel('Number of Tracked Updates (log scale)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    
    # Add title with model and test information
    title = "Accuracy vs Number of Tracked Updates"
    if model_name:
        title += f"\nModel: {model_name}"
    if test_name:
        title += f"\nTest: {test_name}"
    if random_update is not None:
        title += f"\nRandom Updates: {'On' if random_update else 'Off'}"
    plt.title(title, fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Plot saved as: {output_file}")
    
    # 添加模型名称和ci95计算方法到DataFrame
    df['model_name'] = model_name if model_name else 'Unknown'
    df['ci95_method'] = 'Bootstrap (n=5000)'
    
    return df

def extract_model_from_path(path):
    """Extract model name from the path"""
    # Split the path and look for common model name patterns
    path_parts = path.replace('\\', '/').split('/')
    
    # Look for directory that looks like a model name
    for part in reversed(path_parts):
        if part and not part.startswith('eval_pi') and not part.startswith('test_'):
            # Common model name patterns
            if any(term in part.lower() for term in ['gpt', 'gemini', 'claude', 'llama', 'mistral', 'qwen']):
                return part
    
    return None

def extract_test_params_from_path(path):
    """Extract test parameters from the path"""
    path_parts = path.replace('\\', '/').split('/')
    
    test_name = None
    random_update = None
    
    for part in path_parts:
        if part.startswith('test_'):
            test_name = part
            # Check for random update indicators in test name
            if 'randomoff' in part:
                random_update = 0
            elif 'randomon' in part:
                random_update = 1
    
    return test_name, random_update

def collect_data_len_item_mode(base_dir):
    """Collect len_item and accuracy data for test_itemlen tests"""
    data = []
    raw_data_list = []
    
    # Try to determine test name from path for config lookup
    test_name = None
    path_parts = base_dir.replace('\\', '/').split('/')
    for part in path_parts:
        if part.startswith('test_'):
            test_name = part
            break
    
    # Find all test directories with len_item pattern
    pattern = os.path.join(base_dir, "*lenitm-*")
    test_dirs = sorted(glob.glob(pattern))
    
    print(f"Found {len(test_dirs)} test directories for len_item analysis")
    
    for test_dir in test_dirs:
        dir_name = os.path.basename(test_dir)
        len_item = extract_len_item(dir_name)
        n_updates = extract_n_tracked_updates(dir_name)
        
        if len_item is None or n_updates is None:
            continue
            
        # Find the CSV file
        csv_pattern = os.path.join(test_dir, "*_trial.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f"Warning: No CSV file found in {dir_name}")
            continue
            
        csv_file = csv_files[0]
        accuracy = extract_accuracy_from_csv(csv_file)
        
        if accuracy is not None:
            data.append({
                'len_item': len_item,
                'n_tracked_updates': n_updates,
                'accuracy_mean': accuracy['mean'],
                'accuracy_ci95': accuracy['ci95'],
                'n': accuracy['n'],
                'directory': dir_name
            })
            
            # 添加原始数据到列表
            if 'raw_data' in accuracy:
                raw_df = accuracy['raw_data']
                raw_df['len_item'] = len_item
                raw_df['n_tracked_updates'] = n_updates
                raw_df['directory'] = dir_name
                raw_data_list.append(raw_df)
            
            print(f"✅ len_item={len_item:2d}, n_updates={n_updates:2d}, accuracy={accuracy['mean']:.3f} ± {accuracy['ci95']:.3f}")
        else:
            print(f"❌ Failed to extract accuracy from {dir_name}")
    
    # 保存原始数据到CSV文件
    if raw_data_list:
        raw_data_df = pd.concat(raw_data_list, ignore_index=True)
        current_dir = os.getcwd()
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        model_name = os.path.basename(base_dir)
        raw_data_file = os.path.join(current_dir, f'raw_accuracy_data_{test_name}_{model_name}_{timestamp}.csv')
        raw_data_df.to_csv(raw_data_file, index=False)
        print(f"\n📄 Raw data saved as: {raw_data_file}")
    
    return data

def create_len_item_plots(data, base_output_file="accuracy_vs_len_item.png", model_name=None, test_name=None):
    """Create plots showing accuracy vs len_item, separated by n_tracked_updates"""
    if not data:
        print("No data to plot!")
        return []
        
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(data)
    
    # Get unique n_tracked_updates values
    unique_updates = sorted(df['n_tracked_updates'].unique())
    
    result_dfs = []
    
    for n_updates in unique_updates:
        # Filter data for this n_tracked_updates value
        subset_df = df[df['n_tracked_updates'] == n_updates].copy()
        subset_df = subset_df.sort_values('len_item')
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot with error bars
        plt.errorbar(
            subset_df['len_item'],
            subset_df['accuracy_mean'],
            yerr=subset_df['accuracy_ci95'],
            fmt='o-',
            capsize=5,
            markersize=8,
            label='Mean with 95% CI'
        )
        
        # Add some statistics
        max_acc = subset_df['accuracy_mean'].max()
        min_acc = subset_df['accuracy_mean'].min()
        mean_acc = subset_df['accuracy_mean'].mean()
        
        # Add text box with statistics
        stats_text = f'Max Accuracy: {max_acc:.3f}\nMin Accuracy: {min_acc:.3f}\nMean Accuracy: {mean_acc:.3f}\nData Points: {len(subset_df)}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                 verticalalignment='top', fontsize=10)
        
        # **添加对数刻度** - 与其他模式保持一致
        plt.xscale('log')
        
        # Set custom x-axis ticks to show each data point exactly
        plt.xticks(subset_df['len_item'], labels=subset_df['len_item'].astype(str), rotation=45, ha='right')
        plt.xlabel('Item Length (log scale)', fontsize=14)  # 更新标签说明使用了对数刻度
        plt.ylabel('Accuracy', fontsize=14)
        
        # Add title with detailed information
        title = f"Accuracy vs Item Length (n_tracked_updates={n_updates})"
        if model_name:
            title += f"\nModel: {model_name}"
        if test_name:
            title += f"\nTest: {test_name}"
        plt.title(title, fontsize=14)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Generate output filename
        base_name = base_output_file.replace('.png', '')
        output_file = f"{base_name}_n_updates_{n_updates}.png"
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Plot saved as: {output_file}")
        
        # 添加额外信息到DataFrame
        subset_df['model_name'] = model_name if model_name else 'Unknown'
        subset_df['ci95_method'] = 'Bootstrap (n=5000)'
        
        result_dfs.append(subset_df)
    
    return result_dfs

def extract_n_tracked_keys_from_dir(dir_name):
    """Extract n_tracked_keys value from directory name"""
    if "_ntrk-" in dir_name:
        try:
            return int(dir_name.split("_ntrk-")[1].split("_")[0])
        except:
            return None
    return None

def collect_data_nkeys_mode(base_dir):
    """Collect n_tracked_keys and accuracy data for test_nkeys tests"""
    data = []
    raw_data_list = []
    
    # Try to determine test name from path for config lookup
    test_name = None
    path_parts = base_dir.replace('\\', '/').split('/')
    for part in path_parts:
        if part.startswith('test_'):
            test_name = part
            break
    
    # Find all test directories with ntrk pattern
    pattern = os.path.join(base_dir, "*ntrk-*")
    test_dirs = sorted(glob.glob(pattern))
    
    print(f"Found {len(test_dirs)} test directories for n_tracked_keys analysis")
    
    for test_dir in test_dirs:
        dir_name = os.path.basename(test_dir)
        n_tracked_keys = extract_n_tracked_keys_from_dir(dir_name)
        n_updates = extract_n_tracked_updates(dir_name)
        
        if n_tracked_keys is None or n_updates is None:
            continue
            
        # Find the CSV file
        csv_pattern = os.path.join(test_dir, "*_trial.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f"Warning: No CSV file found in {dir_name}")
            continue
            
        csv_file = csv_files[0]
        accuracy = extract_accuracy_from_csv(csv_file)
        
        if accuracy is not None:
            data.append({
                'n_tracked_keys': n_tracked_keys,
                'n_tracked_updates': n_updates,
                'accuracy_mean': accuracy['mean'],
                'accuracy_ci95': accuracy['ci95'],
                'n': accuracy['n'],
                'directory': dir_name
            })
            
            # 添加原始数据到列表
            if 'raw_data' in accuracy:
                raw_df = accuracy['raw_data']
                raw_df['n_tracked_keys'] = n_tracked_keys
                raw_df['n_tracked_updates'] = n_updates
                raw_df['directory'] = dir_name
                raw_data_list.append(raw_df)
            
            print(f"✅ n_tracked_keys={n_tracked_keys:2d}, n_updates={n_updates:2d}, accuracy={accuracy['mean']:.3f} ± {accuracy['ci95']:.3f}")
        else:
            print(f"❌ Failed to extract accuracy from {dir_name}")
    
    # 保存原始数据到CSV文件
    if raw_data_list:
        raw_data_df = pd.concat(raw_data_list, ignore_index=True)
        current_dir = os.getcwd()
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        model_name = os.path.basename(base_dir)
        raw_data_file = os.path.join(current_dir, f'raw_accuracy_data_{test_name}_{model_name}_{timestamp}.csv')
        raw_data_df.to_csv(raw_data_file, index=False)
        print(f"\n📄 Raw data saved as: {raw_data_file}")
    
    return data

def create_nkeys_plots(data, base_output_file="accuracy_vs_nkeys.png", model_name=None, test_name=None):
    """Create plots showing accuracy vs n_tracked_keys, separated by n_tracked_updates"""
    if not data:
        print("No data to plot!")
        return []
        
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(data)
    
    # Get unique n_tracked_updates values
    unique_updates = sorted(df['n_tracked_updates'].unique())
    
    result_dfs = []
    
    for n_updates in unique_updates:
        # Filter data for this n_tracked_updates value
        subset_df = df[df['n_tracked_updates'] == n_updates].copy()
        subset_df = subset_df.sort_values('n_tracked_keys')
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot with error bars
        plt.errorbar(
            subset_df['n_tracked_keys'],
            subset_df['accuracy_mean'],
            yerr=subset_df['accuracy_ci95'],
            fmt='o-',
            capsize=5,
            markersize=8,
            label='Mean with 95% CI'
        )
        
        # Add some statistics
        max_acc = subset_df['accuracy_mean'].max()
        min_acc = subset_df['accuracy_mean'].min()
        mean_acc = subset_df['accuracy_mean'].mean()
        
        # Add text box with statistics
        stats_text = f'Max Accuracy: {max_acc:.3f}\nMin Accuracy: {min_acc:.3f}\nMean Accuracy: {mean_acc:.3f}\nData Points: {len(subset_df)}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                 verticalalignment='top', fontsize=10)
        
        # Use log scale for better visualization (like default mode)
        plt.xscale('log')
        
        # Set custom x-axis ticks to show each data point exactly
        plt.xticks(subset_df['n_tracked_keys'], labels=subset_df['n_tracked_keys'].astype(str), rotation=45, ha='right')
        plt.xlabel('Number of Tracked Keys (log scale)', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        
        # Add title with detailed information
        title = f"Accuracy vs Number of Tracked Keys (n_tracked_updates={n_updates})"
        if model_name:
            title += f"\nModel: {model_name}"
        if test_name:
            title += f"\nTest: {test_name}"
        plt.title(title, fontsize=14)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Generate output filename
        base_name = base_output_file.replace('.png', '')
        output_file = f"{base_name}_n_updates_{n_updates}.png"
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Plot saved as: {output_file}")
        
        # 添加额外信息到DataFrame
        subset_df['model_name'] = model_name if model_name else 'Unknown'
        subset_df['ci95_method'] = 'Bootstrap (n=5000)'
        
        result_dfs.append(subset_df)
    
    return result_dfs

def main():
    parser = argparse.ArgumentParser(description="Plot accuracy vs n_tracked_updates, len_item, or n_tracked_keys")
    
    # Determine default path based on current working directory
    if os.path.basename(os.getcwd()) == "pi_automation":
        default_path = "../eval_pi/test_updates_custom_fast/gemini-2.0-flash"
    else:
        default_path = "eval_pi/test_updates_custom_fast/gemini-2.0-flash"
    
    parser.add_argument("--path", default=default_path,
                       help="Path to test results directory")
    parser.add_argument("--output", default="accuracy_plot.png",
                       help="Output plot filename")
    parser.add_argument("--csv", default="accuracy_summary.csv",
                       help="Output CSV filename")
    
    args = parser.parse_args()
    base_dir = args.path
    
    print("🔍 Analyzing Test Results")
    print("=" * 60)
    print(f"📁 Source: {base_dir}")
    
    if not os.path.exists(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        return
    
    # Extract model and test parameters
    model_name = extract_model_from_path(base_dir)
    test_name, random_update = extract_test_params_from_path(base_dir)
    
    # **检测测试类型** - 按优先级检查
    is_len_item_test = test_name and test_name.startswith('test_itemlen')
    is_nkeys_test = test_name and test_name.startswith('test_nkeys')
    
    if is_len_item_test:
        print("🔬 Detected len_item test - using len_item analysis mode")
        
        # 使用 len_item 模式
        data = collect_data_len_item_mode(base_dir)
        
        if not data:
            print("❌ No data collected!")
            return
        
        print(f"\n📊 Collected {len(data)} data points")
        
        # 创建 len_item 图表
        result_dfs = create_len_item_plots(data, args.output, model_name, test_name)
        
        # 保存CSV文件（每个 n_tracked_updates 一个文件）
        for df in result_dfs:
            n_updates = df['n_tracked_updates'].iloc[0]
            csv_file = args.csv.replace('.csv', f'_n_updates_{n_updates}.csv')
            df.to_csv(csv_file, index=False)
            print(f"📄 Data saved as: {csv_file}")
            
            # 打印摘要表
            print(f"\n📋 Summary Table (n_tracked_updates={n_updates}):")
            print("-" * 30)
            print("len_item | accuracy")
            print("-" * 30)
            for _, row in df.iterrows():
                print(f"{row['len_item']:8d} | {row['accuracy_mean']:.3f} ± {row['accuracy_ci95']:.3f}")
            print("-" * 30)
    
    elif is_nkeys_test:
        print("🔑 Detected n_tracked_keys test - using n_tracked_keys analysis mode")
        
        # 使用 nkeys 模式
        data = collect_data_nkeys_mode(base_dir)
        
        if not data:
            print("❌ No data collected!")
            return
        
        print(f"\n📊 Collected {len(data)} data points")
        
        # 创建 nkeys 图表
        result_dfs = create_nkeys_plots(data, args.output, model_name, test_name)
        
        # 保存CSV文件（每个 n_tracked_updates 一个文件）
        for df in result_dfs:
            n_updates = df['n_tracked_updates'].iloc[0]
            csv_file = args.csv.replace('.csv', f'_n_updates_{n_updates}.csv')
            df.to_csv(csv_file, index=False)
            print(f"📄 Data saved as: {csv_file}")
            
            # 打印摘要表
            print(f"\n📋 Summary Table (n_tracked_updates={n_updates}):")
            print("-" * 35)
            print("n_tracked_keys | accuracy")
            print("-" * 35)
            for _, row in df.iterrows():
                print(f"{row['n_tracked_keys']:13d} | {row['accuracy_mean']:.3f} ± {row['accuracy_ci95']:.3f}")
            print("-" * 35)
    
    else:
        # **完全使用原有的默认行为** - 没有任何更改
        print("📈 Using default n_tracked_updates analysis mode")
        
        # 使用原有的数据收集和绘图函数
        data = collect_data(base_dir)
        
        if not data:
            print("❌ No data collected!")
            return
        
        print(f"\n📊 Collected {len(data)} data points")
        
        # 使用原有的绘图函数
        df = create_plot(data, args.output, model_name, test_name, random_update)
        
        # 打印原有的摘要表
        print("\n📋 Summary Table:")
        print("-" * 40)
        print("n_tracked_updates | accuracy")
        print("-" * 40)
        for _, row in df.iterrows():
            print(f"{row['n_tracked_updates']:16d} | {row['accuracy_mean']:.3f} ± {row['accuracy_ci95']:.3f}")
        print("-" * 40)
        
        # 保存原有格式的CSV
        df.to_csv(args.csv, index=False)
        print(f"📄 Data saved as: {args.csv}")
        print(f"   - Model: {model_name}")
        print(f"   - CI95 Method: Bootstrap (n=5000)")

if __name__ == "__main__":
    main() 