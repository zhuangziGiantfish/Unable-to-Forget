#!/usr/bin/env python3

# Analysis script for pi_flow_final.py results
import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive mode
import matplotlib.pyplot as plt
import seaborn as sns
import re
import scipy.stats as stats
import argparse
import collections
from scipy.stats import pearsonr
from datetime import datetime
import os

# Add argument parsing
parser = argparse.ArgumentParser(description='Analyze PI Flow Final test results')
parser.add_argument('--result_path', type=str, help='Path to the result directory')
parser.add_argument('--parent_dir', action='store_true', help='Treat result_path as parent directory with multiple test configurations')
args = parser.parse_args()

# Function to load JSON files
def load_json(f):
    with open(f, "r", encoding="utf-8") as f:
        return json.load(f)

# Function to calculate 95% confidence interval for a proportion
def calculate_ci95(p, n, alpha=0.05):
    """
    Calculate 95% confidence interval for a proportion using Wilson score interval.
    
    Args:
        p: proportion (accuracy)
        n: sample size
        alpha: significance level (default: 0.05 for 95% CI)
    
    Returns:
        tuple: (lower bound, upper bound, CI width)
    """
    if n == 0:
        return 0, 0, 0
    
    # Wilson score interval (more robust than normal approximation for small n or extreme p)
    z = stats.norm.ppf(1 - alpha/2)
    denominator = 1 + z**2/n
    centre_adjusted_p = (p + z**2/(2*n))/denominator
    adjusted_interval = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))/denominator
    
    lower = max(0, centre_adjusted_p - adjusted_interval)
    upper = min(1, centre_adjusted_p + adjusted_interval)
    ci_width = upper - lower
    
    return lower, upper, ci_width

# Function to estimate required sample size for desired CI width
def estimate_sample_size(p, desired_width, alpha=0.05):
    """
    Estimate sample size needed for a desired confidence interval width.
    
    Args:
        p: current proportion estimate
        desired_width: desired width of confidence interval
        alpha: significance level (default: 0.05 for 95% CI)
    
    Returns:
        int: estimated sample size needed
    """
    z = stats.norm.ppf(1 - alpha/2)
    # Formula based on normal approximation
    n = (4 * z**2 * p * (1-p)) / (desired_width**2)
    return max(30, int(np.ceil(n)))  # Ensure minimum sample size of 30

# Function to analyze a single configuration directory
def analyze_config_directory(project_path, output_parent_dir=None):
    """
    Analyze a single test configuration directory
    
    Args:
        project_path: Path object pointing to the directory containing test results
        output_parent_dir: Optional parent directory to save output files
        
    Returns:
        dict: Analysis results with key metrics
    """
    print(f"\nAnalyzing configuration: {project_path.name}")
    
    # Load result files
    collection_all_pairs = None
    collection_responses = None
    collection_tracked = None
    dict_meta_info = None
    raw_output = None
    collection_forget = None

    # Find subdirectories containing "pi-flow" in the name
    pi_flow_dirs = list(project_path.glob("*pi-flow*"))
    
    # If we have pi-flow subdirectories, analyze each one and aggregate results
    if pi_flow_dirs and args.parent_dir:
        print(f"Found {len(pi_flow_dirs)} pi-flow subdirectories")
        all_pairs_merged = []
        responses_merged = []
        tracked_merged = []
        meta_info = None
        
        for flow_dir in pi_flow_dirs:
            for f in flow_dir.glob("*.json"):
                suffix = f.name.split(".")[0].split("_")[-1]
                if suffix == "all-pairs":
                    data = load_json(f)
                    all_pairs_merged.extend(data)
                elif suffix == "responses":
                    data = load_json(f)
                    responses_merged.extend(data)
                elif suffix == "tracked":
                    data = load_json(f)
                    tracked_merged.extend(data)
                elif suffix == "meta-info":
                    meta_info = load_json(f)
                elif suffix == "raw-output":
                    # Just use the first one we find
                    if raw_output is None:
                        try:
                            with open(f, "r", encoding="utf-8") as file:
                                raw_output = file.read()
                        except:
                            print(f"Error loading raw output from {f}")
        
        if all_pairs_merged and responses_merged and tracked_merged and meta_info:
            collection_all_pairs = all_pairs_merged
            collection_responses = responses_merged
            collection_tracked = tracked_merged
            dict_meta_info = meta_info
            print(f"Aggregated data: {len(collection_all_pairs)} sessions")
    else:
        # Regular single directory analysis
        for f in project_path.glob("*.json"):
            suffix = f.name.split(".")[0].split("_")[-1]
            if suffix == "all-pairs":
                collection_all_pairs = load_json(f)
                print(f"Loaded all-pairs: {len(collection_all_pairs)} sessions")
            elif suffix == "responses":
                collection_responses = load_json(f)
                print(f"Loaded responses: {len(collection_responses)} sessions")
            elif suffix == "tracked":
                collection_tracked = load_json(f)
                print(f"Loaded tracked entries: {len(collection_tracked)} sessions")
            elif suffix == "meta-info":
                dict_meta_info = load_json(f)
                print(f"Loaded meta-info")
                print(f"Available meta-info keys: {dict_meta_info.keys()}")
            elif suffix == "raw-output":
                try:
                    with open(f, "r", encoding="utf-8") as file:
                        raw_output = file.read()
                    print(f"Loaded raw output")
                except:
                    print(f"Error loading raw output")
            elif suffix == "forget":
                collection_forget = load_json(f)
                print(f"Loaded forget entries: {len(collection_forget)} sessions")
    
    # Check if we have all the data we need
    if not all([collection_all_pairs, collection_responses, collection_tracked, dict_meta_info]):
        print(f"Error: Missing some required data files in {project_path}")
        return None
    
    # If no explicit output directory is specified, use the project path
    result_dir = output_parent_dir if output_parent_dir else project_path
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract configuration info from directory name if available
    config_params = {}
    if "tk" in project_path.name and "tu" in project_path.name:
        name_parts = project_path.name.split("_")
        for part in name_parts:
            if part.startswith("tk"):
                config_params["tracked_keys"] = int(part[2:])
            elif part.startswith("uk"):
                config_params["untracked_keys"] = int(part[2:])
            elif part.startswith("tu"):
                config_params["tracked_updates"] = int(part[2:])
            elif part.startswith("uu"):
                config_params["untracked_updates"] = int(part[2:])
            elif part.startswith("mt"):
                config_params["multiply_token"] = int(part[2:])
            elif part.startswith("mf"):
                config_params["multiply_format"] = part[2:]
    
    print("\nAnalyzing position preferences...")
    n_sessions = len(collection_all_pairs)
    
    # Analyze position preferences
    list_resp_pos = []
    list_correct = []
    # --- Store session-level data ---
    session_responses_details = collections.defaultdict(list) 
    # {session_idx: [(pos, is_correct), ...]}
    
    error_details = []  # Track detailed error information
    for idx in range(n_sessions):
        list_pairs = collection_all_pairs[idx]
        response = collection_responses[idx]
        tracked = collection_tracked[idx]
        
        # Separate tracked and untracked entries from all_pairs
        tracked_dict = {}
        for key in tracked:
            tracked_dict[key] = []
        
        # Create case-insensitive lookup for tracked keys
        tracked_lower = {k.lower() if isinstance(k, str) else k: k for k in tracked}
        
        for pair in list_pairs:
            pair_key = pair[0]
            pair_key_lower = pair_key.lower() if isinstance(pair_key, str) else pair_key
            
            # Check if key exists in tracked entries (case-insensitive)
            if pair_key_lower in tracked_lower:
                original_key = tracked_lower[pair_key_lower]
                tracked_dict[original_key].append(pair[1])
        
        # Find position of each response in stream for tracked keys only
        try:
            for k_resp, v_resp in response.items():
                # Only evaluate items that were meant to be tracked
                # Make key matching case-insensitive
                k_resp_lower = k_resp.lower() if isinstance(k_resp, str) else k_resp
                
                # Check if key exists in tracked entries (case-insensitive)
                if k_resp_lower in tracked_lower:
                    # Get the original case key for retrieving from tracked_dict
                    original_key = tracked_lower[k_resp_lower]
                    stream = tracked_dict[original_key]
                    
                    # Important: Use n_tracked_updates from meta info
                    n_tracked_updates = dict_meta_info["n_tracked_updates"]
                    
                    try:
                        # For number mode, we need to calculate the sum of all values
                        token_format = dict_meta_info.get('multiply_token_format', None)
                        if token_format == 'number':
                            # Try to convert all values in stream to integers or floats for summation
                            numeric_values = []
                            for value in stream:
                                try:
                                    # Try to convert value to a number, removing commas if present
                                    value_clean = value.replace(',', '') if isinstance(value, str) else value
                                    numeric_value = int(value_clean) if isinstance(value_clean, str) and value_clean.isdigit() else float(value_clean)
                                    numeric_values.append(numeric_value)
                                except (ValueError, TypeError):
                                    print(f"Warning: Could not convert value '{value}' to a number in number mode")
                            
                            # Calculate the sum as the expected answer
                            expected_value = sum(numeric_values) if numeric_values else 0
                            
                            # Try to convert the model's response to a number
                            try:
                                # Remove commas from numbers (e.g., "1,234,567" -> "1234567")
                                v_resp_clean = v_resp.replace(',', '')
                                v_resp_numeric = int(v_resp_clean) if v_resp_clean.isdigit() else float(v_resp_clean)
                                # Check if the model's answer matches the sum
                                is_correct = v_resp_numeric == expected_value
                                list_correct.append(is_correct)
                                # For tracking position, use -2 to indicate summation result
                                pos_resp = -2
                                list_resp_pos.append(pos_resp)
                                print(f"Number mode - Key: {k_resp}, Sum: {expected_value}, Model: {v_resp_numeric}, Correct: {is_correct}")
                            except (ValueError, TypeError):
                                print(f"Error: Model response '{v_resp}' for key '{k_resp}' is not a valid number")
                                is_correct = False
                                list_correct.append(is_correct)
                                pos_resp = -1
                                list_resp_pos.append(pos_resp)
                        else:
                            # Standard mode - find position of response in stream
                            pos_resp = stream.index(v_resp)
                            list_resp_pos.append(pos_resp)
                            
                            # Check if correct based on probe target
                            is_correct = False
                            if dict_meta_info['probe_target'] in ['up-to-date', 'current', 'latest', 'most recent', 'final', 'last']:
                                is_correct = str(v_resp) == str(stream[-1])
                                list_correct.append(is_correct)
                            elif dict_meta_info['probe_target'] == 'first':
                                # For first value - compare with the first update
                                is_correct = str(v_resp) == str(stream[0])
                                list_correct.append(is_correct)
                            elif dict_meta_info['probe_target'] == 'second':
                                # For second value - compare with the second update (index 1 if it exists)
                                if len(stream) > 1:
                                    is_correct = str(v_resp) == str(stream[1])
                                else:
                                    is_correct = False  # Not correct if there's no second update
                                list_correct.append(is_correct)
                            elif dict_meta_info['probe_target'] == 'third':
                                # For third value - compare with the third update (index 2 if it exists)
                                if len(stream) > 2:
                                    is_correct = str(v_resp) == str(stream[2])
                                else:
                                    is_correct = False  # Not correct if there's no third update
                                list_correct.append(is_correct)
                            elif dict_meta_info['probe_target'] == 'fifth':
                                # For fifth value - compare with the fifth update (index 4 if it exists)
                                if len(stream) > 4:
                                    is_correct = str(v_resp) == str(stream[4])
                                else:
                                    is_correct = False  # Not correct if there's no fifth update
                                list_correct.append(is_correct)
                            else:
                                # Handle other probe targets if necessary
                                list_correct.append(False)
                        
                        # Store for session-level analysis
                        session_responses_details[idx].append((pos_resp, is_correct))

                    except ValueError:
                        print(f"Session {idx}: Response '{v_resp}' not found in stream for key '{k_resp}'")
                        error_details.append((idx, k_resp, v_resp))  # Store error details
                        list_resp_pos.append(-1)
                        list_correct.append(False)
                        # Store for session-level analysis
                        session_responses_details[idx].append((-1, False))
                else:
                    print(f"Session {idx}: Key '{k_resp}' not found in tracked entries")
                    error_details.append((idx, k_resp, "Key not in tracked entries"))  # Store error details
        except (AttributeError, TypeError):
            print(f"Session {idx}: Invalid response format - {type(response)}")
            continue
    
    # Calculate accuracy
    accuracy = sum(list_correct) / len(list_correct) if list_correct else 0
    
    # Count matches per session - Refined for session accuracy
    session_accuracies_dict = {}
    for session_idx, details in session_responses_details.items():
        session_correct = sum(1 for _, is_correct in details if is_correct)
        
        # Use expected number of tracked keys from meta_info instead of actual processed keys
        expected_keys = dict_meta_info["n_tracked_keys"]
        
        # Log discrepancy for debugging if found
        actual_processed = len(details)
        if actual_processed != expected_keys:
            print(f"WARNING: Session {session_idx} processed {actual_processed}/{expected_keys} expected keys")
        
        # Calculate accuracy based on expected keys, not just processed keys
        session_accuracies_dict[session_idx] = session_correct / expected_keys

    # Print session-by-session accuracy
    print("\nSession-by-session accuracy:")
    session_accuracies_list = [] # List for correlation
    for session_idx in sorted(session_accuracies_dict.keys()):
        acc = session_accuracies_dict[session_idx]
        session_accuracies_list.append(acc)
        # Find corresponding details to get counts
        details = session_responses_details.get(session_idx, [])
        processed = len(details)
        correct = sum(1 for _, is_correct in details if is_correct)
        expected = dict_meta_info["n_tracked_keys"]
        print(f"Session {session_idx}: {correct}/{expected} = {acc:.2%} (processed {processed}/{expected} keys)")
    
    # Calculate confidence interval for overall accuracy
    n_total = len(list_correct)
    ci_lower, ci_upper, ci_width = calculate_ci95(accuracy, n_total)
    
    # Calculate mean and std of session accuracies
    mean_session_acc = np.mean(session_accuracies_list) if session_accuracies_list else 0
    std_session_acc = np.std(session_accuracies_list) if len(session_accuracies_list) > 1 else 0
    
    # Calculate CI95 for session-level accuracy
    session_ci_lower, session_ci_upper, session_ci_width = calculate_ci95(
        mean_session_acc, len(session_accuracies_list))
    
    # Estimate needed sample size for narrower CI
    desired_width = 0.10  # 10% width
    suggested_sessions = estimate_sample_size(mean_session_acc, desired_width)

    # --- Calculate Error Distribution Entropy (Overall) ---
    error_positions_overall = [pos for pos, correct in zip(list_resp_pos, list_correct) if not correct]
    n_errors_overall = len(error_positions_overall)
    overall_error_entropy = 0.0
    max_entropy = np.log2(dict_meta_info['n_tracked_updates'])  # Maximum possible entropy

    if n_errors_overall > 0:
        error_pos_counts_overall = collections.Counter(error_positions_overall)
        raw_entropy = 0.0
        for pos in error_pos_counts_overall:
            p_i = error_pos_counts_overall[pos] / n_errors_overall
            if p_i > 0: # Avoid log2(0)
                raw_entropy -= p_i * np.log2(p_i)
        # Normalize entropy to [0,1] range
        overall_error_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
    # --- End Overall Error Entropy Calculation ---

    # --- Calculate Session-Level Error Entropy ---
    session_entropies_dict = {}
    for session_idx, details in session_responses_details.items():
        session_error_positions = [pos for pos, is_correct in details if not is_correct]
        n_session_errors = len(session_error_positions)
        session_entropy = 0.0
        if n_session_errors > 1: # Need at least 2 errors for non-zero entropy
            session_error_counts = collections.Counter(session_error_positions)
            raw_entropy = 0.0
            for pos in session_error_counts:
                p_i = session_error_counts[pos] / n_session_errors
                if p_i > 0:
                    raw_entropy -= p_i * np.log2(p_i)
            # Normalize session entropy
            session_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
        session_entropies_dict[session_idx] = session_entropy

    session_entropies_list = [session_entropies_dict[idx] for idx in sorted(session_entropies_dict.keys())]
    # --- End Session-Level Entropy Calculation ---
    
    # --- Calculate Correlation between Session Accuracy and Session Entropy ---
    correlation = None
    p_value = None
    correlation_str = "Not enough data points for correlation."
    # Ensure lists are aligned and have enough data
    if len(session_accuracies_list) == len(session_entropies_list) and len(session_accuracies_list) >= 2:
        try:
            correlation, p_value = pearsonr(session_accuracies_list, session_entropies_list)
            corr_sig = "statistically significant" if p_value < 0.05 else "not statistically significant"
            correlation_str = f"Correlation (r={correlation:.3f}, p={p_value:.3f}) - {corr_sig}"
        except ValueError as e:
            correlation_str = f"Could not calculate correlation: {e}"
    # --- End Correlation Calculation ---

    print("\nConfidence Interval Analysis:")
    print(f"Session-level accuracy: {mean_session_acc:.2%} ± {std_session_acc:.2%} (95% CI: {session_ci_lower:.2%}-{session_ci_upper:.2%}, width: {session_ci_width:.2%})")
    print(f"Current number of sessions: {n_sessions}")
    print(f"Suggested number of sessions for {desired_width:.0%} CI width: {suggested_sessions}")
    print(f"Overall accuracy: {accuracy:.2%} (95% CI: {ci_lower:.2%}-{ci_upper:.2%}, width: {ci_width:.2%})")
    
    # Position breakdown
    position_counts = {}
    for pos in list_resp_pos:
        if pos not in position_counts:
            position_counts[pos] = 0
        position_counts[pos] += 1
    
    # Create visualization
    plt.figure(figsize=(12, 14))  # Much taller figure
    
    # Generate histogram using n_tracked_updates instead of n_updates
    n_updates_to_use = dict_meta_info["n_tracked_updates"]
    
    # Add special positions for summation mode (-2) and not found (-1)
    token_format = dict_meta_info.get('multiply_token_format', None)
    if token_format == 'number':
        # For number mode, include special bin for summation (-2)
        counts, bin_edges = np.histogram(list_resp_pos, bins=np.arange(-2.5, n_updates_to_use+0.5))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Custom labels for the x-axis
        x_tick_labels = ['Sum', 'Not Found'] + [str(i) for i in range(n_updates_to_use)]
    else:
        # Standard mode, only include -1 for "not found"
        counts, bin_edges = np.histogram(list_resp_pos, bins=np.arange(-1.5, n_updates_to_use+0.5))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Custom labels for the x-axis
        x_tick_labels = ['Not Found'] + [str(i) for i in range(n_updates_to_use)]
    
    # Color coding
    colors = ['steelblue'] * len(bin_centers)
    
    # Use a different color for response not found (-1)
    not_found_index = np.where(bin_centers == -1)[0]
    if len(not_found_index) > 0:
        colors[not_found_index[0]] = 'firebrick'
    
    # Use a special color for summation results (-2) if in number mode
    if token_format == 'number':
        sum_index = np.where(bin_centers == -2)[0]
        if len(sum_index) > 0:
            colors[sum_index[0]] = 'forestgreen'
    
    # Plot histogram
    plt.bar(bin_centers, counts, width=0.8, color=colors, align='center')
    
    # Add labels and title
    plt.xlabel("Position in Update Sequence (0=first, n-1=last)", fontsize=14)
    plt.ylabel("Count of Responses", fontsize=14)
    
    # Main title with all crucial parameters and config details
    config_str = ""
    if config_params:
        config_str = f" - Config: {config_params.get('tracked_keys', 'N/A')}tk, {config_params.get('tracked_updates', 'N/A')}tu"
        if 'multiply_token' in config_params:
            config_str += f", MT{config_params['multiply_token']}-{config_params.get('multiply_format', 'N/A')}"
    
    # Add important notes to plot
    title_parts = []
    if token_format == 'number':
        title_parts.append(f"NUMBER MODE: Model calculates sum of all values per key")
    title_parts.append(f"Model: {dict_meta_info.get('model_name', 'Unknown')}")
    if 'temperature' in dict_meta_info:
        title_parts.append(f"Temperature: {dict_meta_info['temperature']}")
    title_parts.append(f"Accuracy: {accuracy:.2%}")
    
    plt.title('\n'.join(title_parts), fontsize=14, pad=20)
    
    # Add configuration details as text annotation
    config_details = [
        f"Tracked Keys: {dict_meta_info.get('n_tracked_keys', 'N/A')}",
        f"Untracked Keys: {dict_meta_info.get('n_untracked_keys', 'N/A')}",
        f"Tracked Updates: {dict_meta_info.get('n_tracked_updates', 'N/A')}",
        f"Untracked Updates: {dict_meta_info.get('n_untracked_updates', 'N/A')}",
        f"Random Update: {dict_meta_info.get('random_update', 'N/A')}",
        f"Balanced Sample: {dict_meta_info.get('balanced_sample', 'N/A')}",
        f"Memory Limit: {dict_meta_info.get('memory_limit', 'N/A')}",
        f"Probe Target: {dict_meta_info.get('probe_target', 'N/A')}",
        f"Token Format: {dict_meta_info.get('multiply_token_format', 'N/A')}",
    ]
    
    # Add special notes for number mode
    if token_format == 'number':
        config_details.append("NOTE: In number mode, evaluation compares model's answer with the sum of all values")
    
    # Display configuration details on the plot
    plt.figtext(0.05, 0.02, '\n'.join(config_details), fontsize=10, va='bottom')
    
    # Add a vertical line at the expected position
    if dict_meta_info['probe_target'] == 'up-to-date' or dict_meta_info['probe_target'] == 'current':
        expected_pos = n_updates_to_use - 1
        plt.axvline(x=expected_pos, color='red', linestyle='--', alpha=0.7, 
                   label=f"Expected position: {expected_pos}")
    elif dict_meta_info['probe_target'] == 'first':
        expected_pos = 0
        plt.axvline(x=expected_pos, color='red', linestyle='--', alpha=0.7, 
                   label=f"Expected position: {expected_pos}")
    elif dict_meta_info['probe_target'] == 'second':
        expected_pos = 1
        plt.axvline(x=expected_pos, color='red', linestyle='--', alpha=0.7, 
                   label=f"Expected position: {expected_pos}")
    elif dict_meta_info['probe_target'] == 'third':
        expected_pos = 2
        plt.axvline(x=expected_pos, color='red', linestyle='--', alpha=0.7, 
                   label=f"Expected position: {expected_pos}")
    elif dict_meta_info['probe_target'] == 'fifth':
        expected_pos = 4
        plt.axvline(x=expected_pos, color='red', linestyle='--', alpha=0.7, 
                   label=f"Expected position: {expected_pos}")
    
    # Add position labels
    if token_format == 'number':
        # For number mode with Sum (-2) and Not Found (-1)
        plt.xticks(range(-2, n_updates_to_use), x_tick_labels, rotation=45, ha='right')
    else:
        # Standard mode with only Not Found (-1)
        plt.xticks(range(-1, n_updates_to_use), x_tick_labels, rotation=45, ha='right')
    
    # Add a legend and grid
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add formatted text box with current session parameters
    current_params = (
        "Current Parameters:\n"
        f"Model: {dict_meta_info['model_name']}\n"
        f"Prompt forgetting: {dict_meta_info['prompt_forgetting']}\n"
        f"Randomness: {dict_meta_info.get('random_update', 'not specified')}\n"
        f"Total sessions: {n_sessions}\n"
        f"Tracked keys: {dict_meta_info['n_tracked_keys']}\n"
        f"Untracked keys: {dict_meta_info['n_untracked_keys']}\n"
        f"Tracked updates: {dict_meta_info['n_tracked_updates']}\n"
        f"Untracked updates: {dict_meta_info['n_untracked_updates']}\n"
        f"Multiply token: {dict_meta_info.get('multiply_token', 'N/A')}\n"
        f"Token format: {dict_meta_info.get('multiply_token_format', 'N/A')}\n"
        f"Balanced sample for multiply token: {dict_meta_info.get('balanced_sample_for_multiply_token', 'N/A')}\n"
        f"Target position: {'last' if dict_meta_info['probe_target'] in ['up-to-date', 'current'] else dict_meta_info['probe_target']}\n"
        f"Note: Analysis only on tracked keys ({dict_meta_info['n_tracked_keys']})"
    )
    
    # Add formatted text box with suggested parameters - prioritize session-level stats
    suggested_params = (
        "Statistical Analysis & Suggestions:\n"
        f"SESSION-LEVEL STATISTICS (PRIMARY):\n"
        f"Session-level accuracy: {mean_session_acc:.2%} ± {std_session_acc:.2%}\n"
        f"Session-level 95% CI: {session_ci_lower:.2%}-{session_ci_upper:.2%}\n"
        f"Session-level CI width: {session_ci_width:.2%}\n"
        f"Suggested sessions for {desired_width:.0%} CI width: {suggested_sessions}\n\n"
        f"OVERALL & ERROR METRICS:\n"
        f"Overall response accuracy: {accuracy:.2%}\n"
        f"(CI: {ci_lower:.2%}-{ci_upper:.2%}, width: {ci_width:.2%})\n"
        f"Normalized Entropy of Error Positions: {overall_error_entropy:.3f} (0-1 scale)\n\n"
        f"SESSION-LEVEL ACCURACY vs ENTROPY:\n"
        f"{correlation_str}"
    )
    
    # Add the text boxes to the plot
    plt.figtext(0.22, -0.01, current_params, fontsize=12, 
                bbox=dict(facecolor='lightblue', alpha=0.5))
    
    plt.figtext(0.57, -0.01, suggested_params, fontsize=12,
                bbox=dict(facecolor='lightgreen', alpha=0.5))
    
    # Call tight_layout() first
    plt.tight_layout()
    
    # Then increase bottom margin
    plt.subplots_adjust(bottom=0.25)
    
    # Extract configuration info for filename
    config_id = ""
    if config_params:
        config_id = f"tk{config_params.get('tracked_keys', 'NA')}_tu{config_params.get('tracked_updates', 'NA')}"
        if 'multiply_token' in config_params:
            config_id += f"_mt{config_params['multiply_token']}"
        if 'multiply_format' in config_params:
            config_id += f"_mf{config_params['multiply_format']}"
    else:
        # Extract unique ID from the project path
        config_id = project_path.name.split('_')[2] if len(project_path.name.split('_')) > 2 else "TEST"
    
    # Create filename based on model name, settings, and unique ID
    model_name = dict_meta_info['model_name'].replace(" ", "_")
    forgetting_type = dict_meta_info['prompt_forgetting'].replace(" ", "_")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    test_nickname = dict_meta_info.get('test_nick_name', 'test')
    # Remove model_name from filename to avoid path errors with model names containing slashes
    filename = f"{timestamp}_{config_id}_{forgetting_type}_{test_nickname}_analysis.png"
    filepath = result_dir / filename
    
    # Save with high DPI for good quality
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nFigure automatically saved to: {filepath}")
    
    # Close the figure to free up resources
    plt.close()
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Model: {dict_meta_info['model_name']}")
    print(f"Prompt forgetting: {dict_meta_info['prompt_forgetting']}")
    print(f"Total sessions: {n_sessions}")
    print(f"Tracked keys: {dict_meta_info['n_tracked_keys']}")
    print(f"Untracked keys: {dict_meta_info['n_untracked_keys']}")
    print(f"Tracked updates: {dict_meta_info['n_tracked_updates']}")
    print(f"Untracked updates: {dict_meta_info['n_untracked_updates']}")
    print(f"Multiply token: {dict_meta_info.get('multiply_token', 'N/A')}")
    print(f"Token format: {dict_meta_info.get('multiply_token_format', 'N/A')}")
    print(f"Target position: {'last' if dict_meta_info['probe_target'] in ['up-to-date', 'current'] else dict_meta_info['probe_target']}")
    print(f"Session-level accuracy: {mean_session_acc:.2%} (95% CI: {session_ci_lower:.2%}-{session_ci_upper:.2%}, width: {session_ci_width:.2%})")
    print(f"Suggested sessions for {desired_width:.0%} CI width: {suggested_sessions}")
    print(f"Overall accuracy: {accuracy:.2%} (95% CI: {ci_lower:.2%}-{ci_upper:.2%}, width: {ci_width:.2%})")
    print(f"Normalized Entropy of Error Positions: {overall_error_entropy:.3f} (0-1 scale)")
    print(f"Note: Analysis only considers tracked keys")
    
    # Print session correlation
    print(f"\nSession Accuracy vs. Session Error Entropy Correlation:")
    print(correlation_str)

    # Position breakdown
    print("\nPosition breakdown:")
    total_responses = len(list_resp_pos)
    for pos in sorted(position_counts.keys()):
        count = position_counts[pos]
        percentage = count / total_responses * 100
        if pos == -1:
            pos_name = "Not found"
        elif pos == 0:
            pos_name = "First"
        elif pos == dict_meta_info['n_tracked_updates'] - 1:
            pos_name = "Last"
        else:
            pos_name = f"Position {pos}"
        print(f"{pos_name}: {count} responses ({percentage:.1f}%)")

    # --- Generate Text Report --- 
    report_filename = f"{timestamp}_{config_id}_{forgetting_type}_{test_nickname}_report.txt"
    report_filepath = result_dir / report_filename

    with open(report_filepath, 'w', encoding='utf-8') as f:
        f.write(f"Analysis Report for: {project_path.name}\n")
        f.write("="*30 + "\n")
        f.write(f"Model: {dict_meta_info['model_name']}\n")
        f.write(f"Prompt forgetting: {dict_meta_info['prompt_forgetting']}\n")
        f.write(f"Test Nickname: {test_nickname}\n")
        f.write(f"Randomness: {dict_meta_info.get('random_update', 'not specified')}\n")
        f.write(f"Target position: {'last' if dict_meta_info['probe_target'] in ['up-to-date', 'current'] else dict_meta_info['probe_target']}\n")
        f.write(f"Response format: {dict_meta_info.get('response_format', 'not specified')}\n")
        f.write(f"Prompt updating: {dict_meta_info.get('prompt_updating', 'not specified')}\n")
        
        # Add more details about the experimental settings
        f.write("-"*30 + "\nExperimental Parameters:\n")
        f.write(f"Total sessions: {n_sessions}\n")
        f.write(f"Tracked keys: {dict_meta_info['n_tracked_keys']}\n")
        f.write(f"Untracked keys: {dict_meta_info['n_untracked_keys']}\n")
        f.write(f"Tracked updates: {dict_meta_info['n_tracked_updates']}\n")
        f.write(f"Untracked updates: {dict_meta_info['n_untracked_updates']}\n")
        f.write(f"Multiply token: {dict_meta_info.get('multiply_token', 'N/A')}\n")
        f.write(f"Token format: {dict_meta_info.get('multiply_token_format', 'N/A')}\n")
        f.write(f"Balanced sample for multiply token: {dict_meta_info.get('balanced_sample_for_multiply_token', 'N/A')}\n")
        f.write(f"Balanced sample: {dict_meta_info.get('balanced_sample', 'N/A')}\n")
        f.write(f"Sample replacement: {dict_meta_info.get('sample_replacement', 'N/A')}\n")
        f.write(f"Memory limit: {dict_meta_info.get('memory_limit', 'N/A')}\n")
        f.write(f"Remix category: {dict_meta_info.get('remix_category', 'N/A')}\n")
        f.write(f"Temperature: {dict_meta_info.get('temperature', 'N/A')}\n")
        f.write(f"Max tokens: {dict_meta_info.get('max_tokens', 'N/A')}\n")
        f.write(f"Source dict path: {dict_meta_info.get('source_dict_path', 'N/A')}\n")
        f.write(f"Note: Analysis only considers tracked keys\n")
        
        # Detailed analysis results
        f.write("-"*30 + "\nDetailed Analysis Results:\n")
        f.write(f"Overall accuracy: {accuracy:.2%} (95% CI: {ci_lower:.2%}-{ci_upper:.2%}, width: {ci_width:.2%})\n")
        f.write(f"Normalized Error Entropy: {overall_error_entropy:.3f} (0-1 scale)\n")
        f.write(f"Current number of sessions: {n_sessions}\n")
        f.write(f"Suggested number of sessions for {desired_width:.0%} CI width: {suggested_sessions}\n\n")
        
        # Session-Level Analysis
        f.write("-"*30 + "\nSession-Level Analysis:\n")
        f.write(f"Mean session accuracy: {mean_session_acc:.2%} ± {std_session_acc:.2%}\n")
        f.write(f"Session accuracy 95% CI: {session_ci_lower:.2%}-{session_ci_upper:.2%} (width: {session_ci_width:.2%})\n")
        f.write(f"Correlation (Accuracy vs. Normalized Error Entropy): {correlation_str}\n\n")
        
        # Session-by-session details
        f.write("-"*30 + "\nSession-by-Session Details:\n")
        for session_idx in sorted(session_accuracies_dict.keys()):
            acc = session_accuracies_dict[session_idx]
            entropy = session_entropies_dict[session_idx]
            details = session_responses_details.get(session_idx, [])
            total = len(details)
            correct = sum(1 for _, is_correct in details if is_correct)
            f.write(f"Session {session_idx}: {correct}/{total} = {acc:.2%}, Entropy: {entropy:.3f}\n")
        f.write("\n")
        
        # Per-Session Data
        f.write("-"*30 + "\nPer-Session Data:\n")
        f.write("Session | Accuracy | Normalized Error Entropy (0-1)\n")
        f.write("--------|----------|---------------------\n")
        for idx in sorted(session_accuracies_dict.keys()):
            acc = session_accuracies_dict[idx]
            entropy = session_entropies_dict[idx]
            f.write(f"{idx:<7} | {acc:<8.2%} | {entropy:<.3f}\n")
        
        # Position breakdown
        f.write("-"*30 + "\nPosition Breakdown (Overall):\n")
        total_responses = len(list_resp_pos)
        sorted_pos_counts = sorted(position_counts.items())
        for pos, count in sorted_pos_counts:
            percentage = count / total_responses * 100
            if pos == -1: pos_name = "Not found"
            elif pos == 0: pos_name = "First"
            elif pos == dict_meta_info['n_tracked_updates'] - 1: pos_name = "Last"
            else: pos_name = f"Position {pos}"
            f.write(f"{pos_name}: {count} responses ({percentage:.1f}%)\n")
        
        # Detailed errors
        if error_details:
            f.write("-"*30 + "\nDetailed Response Errors:\n")
            for session_idx, key, value in error_details:
                f.write(f"Session {session_idx}: Response '{value}' not found in stream for key '{key}'\n")

        # Special note for number mode
        if token_format == 'number':
            f.write(f"\nNOTE: Using NUMBER MODE - Evaluating by comparing model's answer with sum of all values\n")
            
            # Add a summary of correct sums for the last session
            if n_sessions > 0:
                last_session_idx = n_sessions - 1
                list_pairs = collection_all_pairs[last_session_idx]
                tracked = collection_tracked[last_session_idx]
                
                f.write(f"\nSummary of correct summations for session {last_session_idx}:\n")
                
                # Recreate tracked_dict for the last session
                tracked_dict = {}
                for key in tracked:
                    tracked_dict[key] = []
                
                for pair in list_pairs:
                    if pair[0] in tracked:
                        tracked_dict[pair[0]].append(pair[1])
                
                # Calculate sums for each key
                for key, stream in tracked_dict.items():
                    numeric_values = []
                    for value in stream:
                        try:
                            # Try to convert value to a number, removing commas if present
                            value_clean = value.replace(',', '') if isinstance(value, str) else value
                            numeric_value = int(value_clean) if isinstance(value_clean, str) and value_clean.isdigit() else float(value_clean)
                            numeric_values.append(numeric_value)
                        except (ValueError, TypeError):
                            pass
                    
                    correct_sum = sum(numeric_values) if numeric_values else 0
                    f.write(f"  - Key: {key}, Correct Sum: {correct_sum}\n")
        
        f.write(f"\nResults:\n")

    print(f"\nDetailed report saved to: {report_filepath}")
    # --- End Text Report ---
    
    # Return key metrics for comparative analysis
    return {
        "config_name": project_path.name,
        "config_params": config_params,
        "accuracy": mean_session_acc,
        "accuracy_ci_lower": session_ci_lower,
        "accuracy_ci_upper": session_ci_upper,
        "ci_width": session_ci_width,
        "error_entropy": overall_error_entropy,
        "n_sessions": n_sessions,
        "correlation": correlation if correlation is not None else 0,
        "position_counts": position_counts,
        "tracked_keys": dict_meta_info['n_tracked_keys'],
        "untracked_keys": dict_meta_info['n_untracked_keys'],
        "tracked_updates": dict_meta_info['n_tracked_updates'],
        "untracked_updates": dict_meta_info['n_untracked_updates'],
        "model": dict_meta_info['model_name'],
        "forgetting": dict_meta_info['prompt_forgetting'],
        "randomness": dict_meta_info.get('random_update', 'not specified'),
    }

# Add new function for comparative analysis
def generate_comparative_analysis(config_results, output_dir):
    """
    Generate comparative analysis and visualizations across multiple test configurations
    
    Args:
        config_results: List of dictionaries with analysis results from each configuration
        output_dir: Directory to save the output files
    """
    print(f"\nGenerating comparative analysis for {len(config_results)} configurations")
    
    if not config_results:
        print("No configuration results to analyze")
        return
    
    # Create a pandas DataFrame for analysis
    import pandas as pd
    
    # Extract key metrics for comparison
    comparison_data = []
    for result in config_results:
        # Extract tracked updates and tracked keys from config_params if available
        tracked_updates = result.get('config_params', {}).get('tracked_updates', result.get('tracked_updates'))
        tracked_keys = result.get('config_params', {}).get('tracked_keys', result.get('tracked_keys'))
        
        # Extract multiply token parameters if available
        multiply_token = result.get('config_params', {}).get('multiply_token', 1)
        multiply_format = result.get('config_params', {}).get('multiply_format', 'none')
        
        # Calculate total updates
        total_updates = tracked_keys * tracked_updates
        if 'untracked_keys' in result and 'untracked_updates' in result:
            total_updates += result['untracked_keys'] * result['untracked_updates']
        
        comparison_data.append({
            'config_name': result['config_name'],
            'tracked_keys': tracked_keys,
            'tracked_updates': tracked_updates,
            'accuracy': result['accuracy'],
            'ci_lower': result['accuracy_ci_lower'],
            'ci_upper': result['accuracy_ci_upper'],
            'ci_width': result['ci_width'],
            'error_entropy': result['error_entropy'],
            'n_sessions': result['n_sessions'],
            'correlation': result['correlation'],
            'total_updates': total_updates,
            'multiply_token': multiply_token,
            'multiply_format': multiply_format,
            'updates_per_key': tracked_updates,  # Alias for plotting
        })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV for further analysis
    csv_path = output_dir / 'configuration_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison data to {csv_path}")
    
    # Generate visualizations
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    sns.set(style="whitegrid")
    
    # 1. Accuracy vs Tracked Updates
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with error bars for CI
    plt.errorbar(df['tracked_updates'], df['accuracy']*100, 
                 yerr=[(df['accuracy']-df['ci_lower'])*100, (df['ci_upper']-df['accuracy'])*100],
                 fmt='o', markersize=8, capsize=6, label='Accuracy with 95% CI')
    
    # Add best fit line
    if len(df) > 1:
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(df['tracked_updates'], df['accuracy']*100)
            x_range = np.linspace(df['tracked_updates'].min(), df['tracked_updates'].max(), 100)
            plt.plot(x_range, intercept + slope*x_range, 'r--', 
                    label=f'Best fit (r²={r_value**2:.3f}, p={p_value:.3f})')
        except:
            pass
    
    # Add labels
    plt.xlabel('Number of Tracked Updates per Key', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Accuracy vs. Number of Tracked Updates', fontsize=16)
    
    # Add data labels
    for i, row in df.iterrows():
        plt.annotate(f"{row['tracked_keys']}tk", 
                    (row['tracked_updates'], row['accuracy']*100),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.grid(True)
    plt.legend()
    
    # Save the figure
    accuracy_vs_updates_path = output_dir / 'accuracy_vs_tracked_updates.png'
    plt.savefig(accuracy_vs_updates_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Entropy vs Tracked Updates
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='tracked_updates', y='error_entropy', data=df, s=100)
    
    # Add best fit line for entropy
    if len(df) > 1:
        try:
            slope, intercept, r_value, p_value, std_err = linregress(df['tracked_updates'], df['error_entropy'])
            x_range = np.linspace(df['tracked_updates'].min(), df['tracked_updates'].max(), 100)
            plt.plot(x_range, intercept + slope*x_range, 'r--', 
                    label=f'Best fit (r²={r_value**2:.3f}, p={p_value:.3f})')
        except:
            pass
    
    plt.xlabel('Number of Tracked Updates per Key', fontsize=14)
    plt.ylabel('Normalized Error Entropy (0-1 scale)', fontsize=14)
    plt.title('Normalized Error Distribution Entropy vs. Number of Tracked Updates', fontsize=16)
    
    # Add data labels
    for i, row in df.iterrows():
        plt.annotate(f"{row['tracked_keys']}tk", 
                    (row['tracked_updates'], row['error_entropy']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.grid(True)
    plt.legend()
    
    # Save the figure
    entropy_vs_updates_path = output_dir / 'entropy_vs_tracked_updates.png'
    plt.savefig(entropy_vs_updates_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Combined visualization with all configurations
    plt.figure(figsize=(14, 10))
    
    # Create a subplot grid
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1])
    
    # Accuracy vs Updates
    ax1 = plt.subplot(gs[0, 0])
    sns.scatterplot(x='tracked_updates', y='accuracy', data=df, ax=ax1, s=80)
    ax1.set_xlabel('Tracked Updates per Key')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs. Tracked Updates')
    
    # Entropy vs Updates
    ax2 = plt.subplot(gs[0, 1])
    sns.scatterplot(x='tracked_updates', y='error_entropy', data=df, ax=ax2, s=80)
    ax2.set_xlabel('Tracked Updates per Key')
    ax2.set_ylabel('Normalized Error Entropy (0-1 scale)')
    ax2.set_title('Normalized Error Entropy vs. Tracked Updates')
    
    # CI Width vs Accuracy
    ax3 = plt.subplot(gs[1, 0])
    sns.scatterplot(x='accuracy', y='ci_width', data=df, ax=ax3, s=80)
    ax3.set_xlabel('Accuracy')
    ax3.set_ylabel('Confidence Interval Width')
    ax3.set_title('CI Width vs. Accuracy')
    
    # Number of sessions vs Config
    ax4 = plt.subplot(gs[1, 1])
    session_bars = sns.barplot(x=df['config_name'], y=df['n_sessions'], ax=ax4)
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('Number of Sessions')
    ax4.set_title('Number of Sessions per Configuration')
    ax4.tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    
    # Save the figure
    combined_viz_path = output_dir / 'combined_comparison.png'
    plt.savefig(combined_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Bubble plot for a 3D view (Updates, Keys, Accuracy)
    if 'tracked_keys' in df.columns and len(df['tracked_keys'].unique()) > 1:
        plt.figure(figsize=(12, 10))
        sizes = df['n_sessions'] * 20  # Scale marker size by number of sessions
        
        scatter = plt.scatter(df['tracked_updates'], df['tracked_keys'], 
                              s=sizes, c=df['accuracy'], cmap='viridis', 
                              alpha=0.7, edgecolors='black')
        
        plt.colorbar(scatter, label='Accuracy')
        
        plt.xlabel('Tracked Updates per Key', fontsize=14)
        plt.ylabel('Number of Tracked Keys', fontsize=14)
        plt.title('Accuracy by Updates and Keys\n(Bubble size represents number of sessions)', fontsize=16)
        
        # Add CI labels
        for i, row in df.iterrows():
            label = f"{row['accuracy']*100:.1f}% (±{row['ci_width']*100/2:.1f})"
            plt.annotate(label, 
                        (row['tracked_updates'], row['tracked_keys']),
                        xytext=(5, 5), textcoords='offset points')
        
        # Save the figure
        bubble_plot_path = output_dir / 'accuracy_bubble_plot.png'
        plt.savefig(bubble_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Generate summary report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    report_path = output_dir / f'{timestamp}_comparative_analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Comparative Analysis Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Configurations Analyzed: {len(df)}\n")
        
        # Best configuration by accuracy
        best_acc_idx = df['accuracy'].idxmax()
        best_acc_config = df.iloc[best_acc_idx]
        
        f.write("\nBest Configuration by Accuracy:\n")
        f.write(f"- Config: {best_acc_config['config_name']}\n")
        f.write(f"- Tracked Keys: {best_acc_config['tracked_keys']}\n")
        f.write(f"- Tracked Updates per Key: {best_acc_config['tracked_updates']}\n")
        f.write(f"- Accuracy: {best_acc_config['accuracy']*100:.2f}% (95% CI: {best_acc_config['ci_lower']*100:.2f}%-{best_acc_config['ci_upper']*100:.2f}%)\n")
        f.write(f"- Normalized Error Entropy: {best_acc_config['error_entropy']:.3f} (0-1 scale)\n")
        f.write(f"- Number of Sessions: {best_acc_config['n_sessions']}\n")
        
        # Narrowest CI configuration
        narrowest_ci_idx = df['ci_width'].idxmin()
        narrowest_ci_config = df.iloc[narrowest_ci_idx]
        
        f.write("\nConfiguration with Narrowest Confidence Interval:\n")
        f.write(f"- Config: {narrowest_ci_config['config_name']}\n")
        f.write(f"- CI Width: {narrowest_ci_config['ci_width']*100:.2f}%\n")
        f.write(f"- Accuracy: {narrowest_ci_config['accuracy']*100:.2f}% (95% CI: {narrowest_ci_config['ci_lower']*100:.2f}%-{narrowest_ci_config['ci_upper']*100:.2f}%)\n")
        f.write(f"- Number of Sessions: {narrowest_ci_config['n_sessions']}\n")
        
        # Correlation Analysis
        if len(df) > 2:
            f.write("\nCorrelation Analysis:\n")
            
            # Calculate correlations between parameters
            corr_matrix = df[['tracked_keys', 'tracked_updates', 'accuracy', 'error_entropy', 'ci_width']].corr()
            
            f.write("Correlation Matrix:\n")
            f.write(f"{corr_matrix.to_string()}\n\n")
            
            # Summarize key correlations
            f.write("Key Correlations:\n")
            acc_tu_corr = corr_matrix.loc['accuracy', 'tracked_updates']
            acc_tk_corr = corr_matrix.loc['accuracy', 'tracked_keys'] if 'tracked_keys' in corr_matrix.index else float('nan')
            ent_tu_corr = corr_matrix.loc['error_entropy', 'tracked_updates']
            
            f.write(f"- Accuracy vs Tracked Updates: {acc_tu_corr:.3f}\n")
            if not np.isnan(acc_tk_corr):
                f.write(f"- Accuracy vs Tracked Keys: {acc_tk_corr:.3f}\n")
            f.write(f"- Normalized Error Entropy vs Tracked Updates: {ent_tu_corr:.3f}\n")
        
        # All configurations summary
        f.write("\nAll Configurations Summary:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Config Name':<30} | {'Tracked Keys':<12} | {'Updates/Key':<11} | {'Accuracy':<15} | {'CI Width':<8} | {'Sessions':<8}\n")
        f.write("-"*60 + "\n")
        
        for _, row in df.sort_values('accuracy', ascending=False).iterrows():
            f.write(f"{row['config_name']:<30} | {row['tracked_keys']:<12} | {row['tracked_updates']:<11} | ")
            f.write(f"{row['accuracy']*100:.2f}% (±{row['ci_width']*100/2:.1f}) | {row['ci_width']*100:.2f}% | {row['n_sessions']:<8}\n")
        
        f.write("\n")
        f.write("Note: The analyses above focus on tracked items only. CI = Confidence Interval (95%).\n")
    
    print(f"Comparative analysis report saved to {report_path}")
    print(f"Generated {len(df)} comparison visualizations")
    
    return df

# Define paths - handle parent directory or single directory
if args.result_path:
    base_path = Path(args.result_path)
    print(f"Using provided path: {base_path}")
    
    if args.parent_dir:
        print(f"Treating as parent directory with multiple test configurations")
        # Find all configuration directories
        config_dirs = [d for d in base_path.iterdir() if d.is_dir() and 
                      (d.name.startswith("tk") or "pi-flow" in d.name)]
        
        if not config_dirs:
            print(f"No test configuration directories found in {base_path}")
            # Try looking in subdirectories
            for subdir in base_path.iterdir():
                if subdir.is_dir():
                    sub_config_dirs = [d for d in subdir.iterdir() if d.is_dir() and 
                                     (d.name.startswith("tk") or "pi-flow" in d.name)]
                    if sub_config_dirs:
                        config_dirs.extend(sub_config_dirs)
        
        if config_dirs:
            print(f"Found {len(config_dirs)} configuration directories")
            # Create output directory for comparative analysis
            output_dir = base_path / "comparative_analysis"
            output_dir.mkdir(exist_ok=True)
            
            # Analyze each configuration
            config_results = []
            for config_dir in config_dirs:
                result = analyze_config_directory(config_dir, output_dir)
                if result:
                    config_results.append(result)
            
            # Generate comparative analysis if we have results
            if config_results:
                print("\nGenerating comparative analysis across all configurations")
                generate_comparative_analysis(config_results, output_dir)
        else:
            print("No configuration directories found, treating as single test directory")
            analyze_config_directory(base_path)
    else:
        # Regular single directory analysis
        analyze_config_directory(base_path)
else:
    # Try to find the most recent pi-flow-final result folder
    try:
        eval_pi_temp_path = Path("../eval_pi_temp")
        if not eval_pi_temp_path.exists():
            eval_pi_temp_path = Path("../../eval_pi_temp")  # try one level up if needed
            
        matching_dirs = list(eval_pi_temp_path.glob("*pi-flow-final*"))
        if matching_dirs:
            # Sort by creation time (most recent first)
            matching_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            project_path = matching_dirs[0]
        else:
            # Look for variable updates directories
            var_update_dirs = list(eval_pi_temp_path.glob("*pi_variable_updates*"))
            if var_update_dirs:
                var_update_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                project_path = var_update_dirs[0]
                print(f"Found variable updates directory: {project_path}")
                args.parent_dir = True  # Treat as parent directory
            else:
                # Fallback to any directory in eval_pi_temp
                all_dirs = [d for d in eval_pi_temp_path.iterdir() if d.is_dir()]
                if all_dirs:
                    all_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    project_path = all_dirs[0]
                else:
                    project_path = eval_pi_temp_path  # Last resort
        
        if args.parent_dir:
            config_dirs = [d for d in project_path.iterdir() if d.is_dir() and 
                          (d.name.startswith("tk") or "pi-flow" in d.name)]
            if config_dirs:
                print(f"Found {len(config_dirs)} configuration directories")
                # Create output directory for comparative analysis
                output_dir = project_path / "comparative_analysis"
                output_dir.mkdir(exist_ok=True)
                
                # Analyze each configuration
                config_results = []
                for config_dir in config_dirs:
                    result = analyze_config_directory(config_dir, output_dir)
                    if result:
                        config_results.append(result)
                
                # Generate comparative analysis if we have results
                if config_results:
                    print("\nGenerating comparative analysis across all configurations")
                    generate_comparative_analysis(config_results, output_dir)
            else:
                print("No configuration directories found, treating as single test directory")
                analyze_config_directory(project_path)
        else:
            analyze_config_directory(project_path)
    except Exception as e:
        print(f"Error finding default directory: {e}")
        project_path = Path("../eval_pi_temp")
        analyze_config_directory(project_path)

if __name__ == "__main__":
    if args.result_path:
        # Path already processed in the above code
        pass
    else:
        # Try to find the most recent pi-flow-final result folder
        try:
            eval_pi_temp_path = Path("../eval_pi_temp")
            if not eval_pi_temp_path.exists():
                eval_pi_temp_path = Path("../../eval_pi_temp")  # try one level up if needed
                
            matching_dirs = list(eval_pi_temp_path.glob("*pi-flow-final*"))
            if matching_dirs:
                # Sort by creation time (most recent first)
                matching_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                project_path = matching_dirs[0]
            else:
                # Look for variable updates directories
                var_update_dirs = list(eval_pi_temp_path.glob("*pi_variable_updates*"))
                if var_update_dirs:
                    var_update_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    project_path = var_update_dirs[0]
                    print(f"Found variable updates directory: {project_path}")
                    args.parent_dir = True  # Treat as parent directory
                else:
                    # Fallback to any directory in eval_pi_temp
                    all_dirs = [d for d in eval_pi_temp_path.iterdir() if d.is_dir()]
                    if all_dirs:
                        all_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                        project_path = all_dirs[0]
                    else:
                        project_path = eval_pi_temp_path  # Last resort
            
            if args.parent_dir:
                config_dirs = [d for d in project_path.iterdir() if d.is_dir() and 
                            (d.name.startswith("tk") or "pi-flow" in d.name)]
                if config_dirs:
                    print(f"Found {len(config_dirs)} configuration directories")
                    # Create output directory for comparative analysis
                    output_dir = project_path / "comparative_analysis"
                    output_dir.mkdir(exist_ok=True)
                    
                    # Analyze each configuration
                    config_results = []
                    for config_dir in config_dirs:
                        result = analyze_config_directory(config_dir, output_dir)
                        if result:
                            config_results.append(result)
                    
                    # Generate comparative analysis if we have results
                    if config_results:
                        print("\nGenerating comparative analysis across all configurations")
                        generate_comparative_analysis(config_results, output_dir)
                else:
                    print("No configuration directories found, treating as single test directory")
                    analyze_config_directory(project_path)
            else:
                analyze_config_directory(project_path)
        except Exception as e:
            print(f"Error finding default directory: {e}")
            project_path = Path("../eval_pi_temp")
            analyze_config_directory(project_path) 