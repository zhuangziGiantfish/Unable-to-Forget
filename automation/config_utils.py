#!/usr/bin/env python3
"""
Utility functions for reading PI test configurations
"""

import yaml
import os

def load_yaml_config(config_path):
    """Load YAML configuration file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config {config_path}: {e}")
        return None

def extract_n_tracked_keys_from_config(config_path):
    """Extract n_tracked_keys from YAML config"""
    config = load_yaml_config(config_path)
    if config and 'parameters' in config and 'n_tracked_keys' in config['parameters']:
        n_tracked_keys = config['parameters']['n_tracked_keys']
        if isinstance(n_tracked_keys, list) and len(n_tracked_keys) > 0:
            return n_tracked_keys[0]  # Take the first value
        elif isinstance(n_tracked_keys, int):
            return n_tracked_keys
    return None

def find_config_for_test(test_name):
    """Find the config file for a given test name"""
    # Common config paths to check for benchmark structure
    possible_paths = [
        f"configs/{test_name}.yaml",
        f"../configs/{test_name}.yaml",
        f"../../configs/{test_name}.yaml",
        f"configs/pi_tests/{test_name}.yaml",
        f"../configs/pi_tests/{test_name}.yaml", 
        f"../../configs/pi_tests/{test_name}.yaml"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def get_n_tracked_keys_for_test(test_name):
    """Get n_tracked_keys value for a specific test"""
    config_path = find_config_for_test(test_name)
    if config_path:
        return extract_n_tracked_keys_from_config(config_path)
    return None

def create_directory_pattern(n_tracked_keys):
    """Create the directory pattern based on n_tracked_keys"""
    return f"*ntrk-{n_tracked_keys}_ntrkupd*" 