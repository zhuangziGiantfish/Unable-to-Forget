#!/usr/bin/env python3
"""
Custom analyzer for PI test results that works with the specific directory structure
created by our test_updates_custom configuration.

This handles directories like: 2025-06-01_22-46-07_ntrk-15_ntrkupd-2/
"""

import os
import sys
import glob
import json
import subprocess
from pathlib import Path
from config_utils import get_n_tracked_keys_for_test, create_directory_pattern

def find_test_directories(base_dir, test_name=None):
    """Find all test result directories using dynamic pattern"""
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
    
    return sorted(glob.glob(pattern))

def extract_n_tracked_updates(dir_name):
    """Extract n_tracked_updates value from directory name"""
    if "_ntrkupd-" in dir_name:
        try:
            return int(dir_name.split("_ntrkupd-")[1].split("_")[0])
        except:
            return None
    return None

def analyze_single_directory(test_dir):
    """Run analysis on a single test directory"""
    test_name = os.path.basename(test_dir)
    n_updates = extract_n_tracked_updates(test_name)
    
    print(f"🔬 Analyzing: {test_name}")
    print(f"   n_tracked_updates: {n_updates}")
    
    # Run the analyzer on this specific directory
    # Use relative path for benchmark structure
    analyzer_script = 'automation/analyze_pi_flow_final.py'
    
    cmd = ['python', analyzer_script, '--result_path', test_dir]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"   ✅ Success")
        return True, n_updates
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e.returncode}")
        if e.stderr:
            print(f"   Error: {e.stderr[:200]}...")
        return False, n_updates

def create_comparison_summary(base_dir, results):
    """Create a summary comparing results across different n_tracked_updates"""
    summary_file = os.path.join(base_dir, "comparison_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("PI Test Results Comparison Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write("Configuration:\n")
        f.write("- Model: gemini-2.0-flash\n")
        f.write("- Random Update: 0\n")
        f.write("- n_tracked_keys: 46\n")
        f.write("- Test Type: test_updates_custom\n\n")
        
        f.write("Results by n_tracked_updates:\n")
        f.write("-" * 30 + "\n")
        
        for success, n_updates in sorted(results, key=lambda x: x[1] if x[1] else 0):
            status = "✅ Analyzed" if success else "❌ Failed"
            f.write(f"n_tracked_updates={n_updates:3d}: {status}\n")
        
        f.write(f"\nTotal: {len(results)} test combinations\n")
        successful = sum(1 for success, _ in results if success)
        f.write(f"Successful analyses: {successful}/{len(results)}\n")
    
    print(f"\n📊 Summary saved to: {summary_file}")

def main():
    # Allow custom path via command line argument
    if len(sys.argv) > 1:
        # Check for help flag
        if sys.argv[1] in ['-h', '--help', 'help']:
            print("🔬 Custom PI Test Analyzer")
            print("=" * 50)
            print("Usage:")
            print("  python analyze_pi_custom.py [path_to_results]")
            print()
            print("Arguments:")
            print("  path_to_results    Path to test results directory")
            print("                     (default: eval_pi/test_updates_custom_fast/gemini-2.0-flash)")
            print()
            print("Examples:")
            print("  python analyze_pi_custom.py")
            print("  python analyze_pi_custom.py eval_pi/my_test/gemini-2.0-flash")
            print("  python analyze_pi_custom.py /path/to/specific/test/directory")
            return 0
        
        base_dir = sys.argv[1]
        print(f"🎯 Using custom path: {base_dir}")
    else:
        # Determine default path based on current working directory
        if os.path.basename(os.getcwd()) == "pi_automation":
            base_dir = "../eval_pi/test_updates_custom_fast/gemini-2.0-flash"
        else:
            base_dir = "eval_pi/test_updates_custom_fast/gemini-2.0-flash"
        print(f"🎯 Using default path: {base_dir}")
    
    print("🔬 Custom PI Test Analyzer")
    print("=" * 50)
    
    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        print("Make sure tests have been run and completed.")
        sys.exit(1)
    
    # Extract test name from path for config lookup
    test_name = None
    path_parts = base_dir.replace('\\', '/').split('/')
    for part in path_parts:
        if part.startswith('test_'):
            test_name = part
            break
    
    # Find test directories with dynamic pattern
    test_dirs = find_test_directories(base_dir, test_name)
    
    if not test_dirs:
        print(f"❌ No test result directories found in {base_dir}")
        if test_name:
            n_tracked_keys = get_n_tracked_keys_for_test(test_name)
            if n_tracked_keys:
                expected_pattern = create_directory_pattern(n_tracked_keys)
                print(f"Expected directories with pattern: {expected_pattern}")
            else:
                print("Could not determine expected pattern from config")
        else:
            print("Expected directories with pattern: *ntrk-<N>_ntrkupd*")
        sys.exit(1)
    
    print(f"📊 Found {len(test_dirs)} test directories")
    
    # Show what we found
    print("\n📋 Test directories:")
    for i, test_dir in enumerate(test_dirs, 1):
        dir_name = os.path.basename(test_dir)
        n_updates = extract_n_tracked_updates(dir_name)
        print(f"  {i:2d}. n_tracked_updates={n_updates:3d} - {dir_name}")
    
    print(f"\n🚀 Running individual analyses...")
    print("-" * 50)
    
    # Analyze each directory
    results = []
    for i, test_dir in enumerate(test_dirs, 1):
        print(f"\n[{i}/{len(test_dirs)}]", end=" ")
        success, n_updates = analyze_single_directory(test_dir)
        results.append((success, n_updates))
    
    # Create summary
    print("\n" + "=" * 50)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 50)
    
    successful = sum(1 for success, _ in results if success)
    print(f"Successful analyses: {successful}/{len(results)}")
    
    if successful > 0:
        print(f"\n✅ Analysis completed!")
        print(f"📁 Individual results saved in each test directory")
        print(f"📄 Look for generated .png, .txt, and .csv files")
        
        # Create comparison summary
        create_comparison_summary(base_dir, results)
        
        print(f"\n🔍 To view all generated files:")
        print(f"   find {base_dir} -name '*.png' -o -name '*.txt' -o -name '*.csv'")
        
    else:
        print(f"\n❌ All analyses failed!")
        print(f"💡 Check if the test results contain the required files:")
        print(f"   - *meta-info.json")
        print(f"   - *all-pairs.json") 
        print(f"   - *responses.json")
        print(f"   - *tracked.json")

if __name__ == "__main__":
    main() 