#!/usr/bin/env python3
"""
Enhanced PI Benchmark Runner (Standalone)
Supports any model without pre-configuration
"""

import subprocess
import sys
import argparse
import yaml
import os
from pathlib import Path

# Default 5 core tests - always available
DEFAULT_TESTS = {
    'test1': 'test1_updates',
    'test2': 'test2_nkeys', 
    'test3': 'test3_ntracked',
    'test4': 'test4_itemlen',
    'test5': 'test5_updates_randomoff'
}

def modify_test_config(test_name, mode='ci95', n_sessions=None):
    """
    Temporarily modify test configuration based on mode
    """
    config_file = f"configs/{test_name}.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ Test configuration not found: {config_file}")
        return None
    
    # Read original config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify n_sessions based on mode
    if mode == 'ci95':
        config['parameters']['n_sessions'] = 0  # CI95 mode
    elif mode == 'quick':
        config['parameters']['n_sessions'] = n_sessions or 5  # Quick mode
    
    # Write temporary config
    temp_config_file = f"configs/{test_name}_temp.yaml"
    with open(temp_config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return temp_config_file

def cleanup_temp_configs():
    """Clean up temporary configuration files"""
    import glob
    temp_files = glob.glob("configs/*_temp.yaml")
    for file in temp_files:
        try:
            os.remove(file)
        except:
            pass

def check_model_in_mapping(model_name):
    """Check if model exists in mapping file (optional)"""
    mapping_file = "configs/model_test_mapping.yaml"
    if not os.path.exists(mapping_file):
        return False, []
    
    try:
        with open(mapping_file, 'r') as f:
            mapping = yaml.safe_load(f)
        
        if model_name in mapping:
            return True, mapping[model_name]
        else:
            return False, []
    except:
        return False, []

def run_test(model, test_name, mode='ci95', n_sessions=None, additional_args=None):
    """Run a single test with specified mode"""
    
    # Create temporary config with modified n_sessions
    temp_config = modify_test_config(test_name, mode, n_sessions)
    if temp_config is None:
        return False
        
    temp_test_name = os.path.basename(temp_config).replace('.yaml', '')
    
    print(f"🧪 Running {test_name} in {mode} mode (n_sessions={'auto' if mode=='ci95' else n_sessions or 5})")
    
    # Build command - use relative paths for core scripts
    cmd = ["python", "automation/run_full_pipeline.py", 
           "--model", model, 
           "--test", temp_test_name]
    
    if additional_args:
        cmd.extend(additional_args)
    
    # Run the test
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {test_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {test_name} failed with exit code {e.returncode}")
        return False
    finally:
        # Clean up temp config
        try:
            os.remove(temp_config)
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description='Enhanced PI Benchmark Runner (Standalone)')
    parser.add_argument('--model', '-m', required=True, help='Model name to test (any name accepted)')
    parser.add_argument('--tests', '-t', default='all', 
                       help='Tests to run: all, test1, test2, test3, test4, test5, or comma-separated list')
    parser.add_argument('--mode', choices=['ci95', 'quick'], default='ci95',
                       help='Test mode: ci95 (robust, longer) or quick (faster, preview)')
    parser.add_argument('--n-sessions', type=int, default=5,
                       help='Number of sessions for quick mode (default: 5)')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip analysis step')
    parser.add_argument('--skip-plot', action='store_true', help='Skip plotting step')
    parser.add_argument('--use-mapping', action='store_true', 
                       help='Use model_test_mapping.yaml if available (optional)')
    
    args = parser.parse_args()
    
    print("🔬 Enhanced PI Benchmark Runner (Standalone)")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode} ({'CI95 robust' if args.mode == 'ci95' else f'Quick preview (n_sessions={args.n_sessions})'})")
    
    # Check if model exists in mapping (optional feature)
    model_in_mapping, mapped_tests = check_model_in_mapping(args.model)
    if args.use_mapping and model_in_mapping:
        print(f"📋 Found model in mapping with tests: {mapped_tests}")
        # Filter mapped tests to only include our 5 core tests
        available_mapped_tests = [t for t in mapped_tests if t in DEFAULT_TESTS.values()]
        if available_mapped_tests:
            print(f"🎯 Available core tests from mapping: {available_mapped_tests}")
    else:
        print(f"🆓 Running without mapping constraints (all 5 tests available)")
    
    # Determine which tests to run
    if args.tests == 'all':
        tests_to_run = list(DEFAULT_TESTS.keys())
    else:
        tests_to_run = [t.strip() for t in args.tests.split(',')]
    
    print(f"Tests: {tests_to_run}")
    print("=" * 50)
    
    # Validate test names
    invalid_tests = [t for t in tests_to_run if t not in DEFAULT_TESTS]
    if invalid_tests:
        print(f"❌ Invalid test names: {invalid_tests}")
        print(f"Available tests: {list(DEFAULT_TESTS.keys())}")
        return 1
    
    # Build additional arguments
    additional_args = []
    if args.skip_analysis:
        additional_args.append('--skip-analysis')
    if args.skip_plot:
        additional_args.append('--skip-plot')
    
    # Run tests
    success_count = 0
    total_count = len(tests_to_run)
    
    for test_key in tests_to_run:
        test_name = DEFAULT_TESTS[test_key]
        print(f"\n{'='*60}")
        print(f"Running {test_key}: {test_name}")
        print(f"{'='*60}")
        
        success = run_test(args.model, test_name, args.mode, args.n_sessions, additional_args)
        if success:
            success_count += 1
    
    # Final summary
    print("\n" + "="*60)
    print("🏁 BENCHMARK COMPLETE")
    print("="*60)
    print(f"✅ Successful: {success_count}/{total_count}")
    print(f"❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All tests completed successfully!")
        return 0
    else:
        print("⚠️  Some tests failed. Check logs above.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrupted by user")
        cleanup_temp_configs()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        cleanup_temp_configs()
        sys.exit(1) 