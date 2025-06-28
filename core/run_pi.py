import yaml
import argparse
from pathlib import Path
import subprocess
import sys
import time
import os
import itertools
import json
import glob
import datetime
import io
import contextlib

## pilot paths
# OUTPUT_ROOT = "eval_pi_pilot"
# CONFIG_ROOT = "configs/pi_pilot_tests"
# FAMILIES_FILE = "configs/model_families.json"
# MAPPING_FILE = f"{CONFIG_ROOT}/model_test_mapping.yaml"

### benchmark paths  
OUTPUT_ROOT = "eval_pi"
CONFIG_ROOT = "configs"
FAMILIES_FILE = "configs/model_families.json"
MAPPING_FILE = f"{CONFIG_ROOT}/model_test_mapping.yaml"


# Global variables to store the log filename and buffer for the current run
LOG_FILENAME = None
LOG_BUFFERS = {}  # Dictionary to store logs per model

def get_log_filename():
    """Get the log filename for the current run, generating it if needed"""
    global LOG_FILENAME
    if LOG_FILENAME is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        LOG_FILENAME = f"terminal_log_{timestamp}.txt"
    return LOG_FILENAME

def get_or_create_log_buffer(model_name):
    """Get the log buffer for a specific model, creating it if it doesn't exist"""
    global LOG_BUFFERS
    if model_name not in LOG_BUFFERS:
        # Initialize with run information
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        LOG_BUFFERS[model_name] = [
            f"========================================",
            f"Starting new run for model {model_name} at {timestamp}",
            f"Python version: {sys.version}",
            f"Working directory: {os.getcwd()}",
            f"Command line: {' '.join(sys.argv)}",
            f"========================================"
        ]
    return LOG_BUFFERS[model_name]

def append_to_log_buffer(message, model_name=None):
    """Append a message to the log buffer for the specified model"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    if model_name:
        # Add to specific model buffer
        buffer = get_or_create_log_buffer(model_name)
        buffer.append(log_entry)
    else:
        # Add to all model buffers
        for model_buffer in LOG_BUFFERS.values():
            model_buffer.append(log_entry)
            
    # Original message is still returned
    return message

def save_log_buffer(model_name, test_name):
    """Save the log buffer for a specific model to a file in the appropriate directory"""
    # Create logs directory within the output directory
    logs_dir = os.path.join(OUTPUT_ROOT, test_name, model_name, "logs")
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    # Get filename for this run
    filename = get_log_filename()
    
    # Get the buffer for this model
    buffer = get_or_create_log_buffer(model_name)
    
    # Save to file
    log_path = os.path.join(logs_dir, filename)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(buffer))
    
    return log_path

# Override print function to capture output
original_print = print
def custom_print(*args, **kwargs):
    """Custom print function that captures output to the model-specific log buffer if context is available"""
    message = io.StringIO()
    original_print(*args, file=message, **kwargs)
    message_str = message.getvalue().strip()
    
    # Get the current model name from thread-local context if it exists
    current_model = getattr(custom_print, 'current_model', None)
    
    if current_model:
        # Add to specific model buffer
        buffer = get_or_create_log_buffer(current_model)
        buffer.append(message_str)
    else:
        # Add to all model buffers if no specific model context
        for model_buffer in LOG_BUFFERS.values():
            model_buffer.append(message_str)
    
    original_print(*args, **kwargs)

print = custom_print

def parse_args():
    parser = argparse.ArgumentParser(description='Run model tests')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', '-m', help='Specify the model to run')
    group.add_argument('--family', '-f', help='Specify a model family to run all models in that family')
    parser.add_argument('--test', '-t',
                       help='Specify tests to run, multiple tests separated by commas. If not specified, run all tests for the model')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing results, skipping already completed parameter combinations')
    return parser.parse_args()

def load_mapping():
    """Load model-test mapping file"""
    with open(MAPPING_FILE, 'r') as f:
        return yaml.safe_load(f)

def load_families():
    """Load model families definition file"""
    with open(FAMILIES_FILE, 'r') as f:
        return json.load(f)

def load_test_config(test_name):
    """Load test configuration file"""
    with open(f'{CONFIG_ROOT}/{test_name}.yaml', 'r') as f:
        return yaml.safe_load(f)

def find_existing_parameter_combinations(test_name, model_name):
    """Find existing parameter combinations from meta-info.json files"""
    existing_combinations = []
    output_dir = os.path.join(OUTPUT_ROOT, test_name, model_name)
    
    if not os.path.exists(output_dir):
        return existing_combinations
    
    # Find all meta-info.json files in the output directory and its subdirectories
    meta_files = glob.glob(os.path.join(output_dir, "**", "*meta-info.json"), recursive=True)
    
    for meta_file in meta_files:
        try:
            with open(meta_file, 'r') as f:
                meta_info = json.load(f)
                # Extract only the parameters that are in the meta_info
                params = {k: str(v) for k, v in meta_info.items() 
                         if k not in ['test_name', 'output_dir', 'if_finished', 'model_name']}
                existing_combinations.append(params)
        except Exception as e:
            print(f"Warning: Could not read {meta_file}: {e}")
    
    return existing_combinations

def is_combination_existing(params_dict, existing_combinations):
    """Check if a parameter combination already exists by comparing only the crossed parameters"""
    # Convert all values to strings for consistent comparison
    params_dict = {k: str(v) for k, v in params_dict.items()}
    
    for existing in existing_combinations:
        # Check if all parameters in params_dict exist in existing with the same values
        if all(k in existing and str(existing[k]) == str(v) for k, v in params_dict.items()):
            return True
    return False

def get_abbrev_map():
    """Return a mapping from parameter names to abbreviations."""
    return {
        'model_name': 'mdl',
        'temperature': 'temp',
        'max_tokens': 'maxtok',
        'n_sessions': 'nsess',
        'n_tracked_keys': 'ntrk',
        'n_untracked_keys': 'nuntrk',
        'n_tracked_updates': 'ntrkupd',
        'n_untracked_updates': 'nuntrkupd',
        'random_update': 'rndupd',
        'balanced_sample': 'balsmpl',
        'memory_limit': 'memlim',
        'probe_target': 'probe',
        'prompt_updating': 'updstyle',
        'prompt_forgetting': 'frgstyle',
        'response_format': 'respfmt',
        'sample_replacement': 'repl',
        'remix_category': 'remix',
        'lengthen_item': 'lenitm',
        'hack_track_keys': 'hacktrk',
        'len_item': 'lenitm',
    }

def make_test_nick_name(params, abbrev_map, crossed_keys):
    """Generate test_nick_name from crossed parameters using abbreviations."""
    parts = []
    for k in crossed_keys:
        if k in params:
            value = str(params[k])
            ## replace all '_' with '&'
            value = value.replace('_', '&')
            parts.append(f"{abbrev_map.get(k, k)}-{value}")
    return "_".join(parts)

def generate_parameter_combinations(params, existing_combinations=None):
    """Generate parameter cross combinations, optionally skipping existing ones"""
    cross_params = {k: v for k, v in params.items() if isinstance(v, list)}
    fixed_params = {k: v for k, v in params.items() if not isinstance(v, list)}
    abbrev_map = get_abbrev_map()
    crossed_keys = list(cross_params.keys())

    if cross_params:
        keys = cross_params.keys()
        values = cross_params.values()
        combinations = list(itertools.product(*values))
        all_combinations = []
        for combo in combinations:
            params_dict = dict(zip(keys, combo))
            if existing_combinations and is_combination_existing(params_dict, existing_combinations):
                print(f"Skipping existing combination: {params_dict}")
                continue
            full_params = fixed_params.copy()
            full_params.update(params_dict)
            # Generate test_nick_name for this combination
            test_nick_name = make_test_nick_name(full_params, abbrev_map, crossed_keys)
            full_params['test_nick_name'] = test_nick_name
            all_combinations.append(full_params)
        return all_combinations
    else:
        if existing_combinations and is_combination_existing(fixed_params, existing_combinations):
            print(f"Skipping existing combination: {fixed_params}")
            return []
        # Even for fixed params, generate test_nick_name (empty or with all fixed params)
        test_nick_name = make_test_nick_name(fixed_params, get_abbrev_map(), [])
        fixed_params['test_nick_name'] = test_nick_name
        return [fixed_params]

def params_to_cmd_args(params):
    """Convert parameter dictionary to command line argument list"""
    cmd_args = []
    for key, value in params.items():
        if value is not None:  # Skip parameters with None value
            cmd_args.extend([f'--{key}', str(value)])
    return cmd_args

def get_models_from_family(family_name, mapping):
    """Get list of models in a family that exist in the mapping"""
    try:
        families = load_families()
        if family_name not in families:
            print(f"Error: Unknown family {family_name}")
            sys.exit(1)
            
        # Filter models in the family that are also in the mapping
        models = [model for model in families[family_name] if model in mapping]
        if not models:
            print(f"Warning: No models from family {family_name} found in the mapping")
        return models
    except FileNotFoundError:
        print(f"Error: Model families file {FAMILIES_FILE} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Model families file {FAMILIES_FILE} is not valid JSON")
        sys.exit(1)

def get_tasks_to_run(mapping, args):
    """Get list of tasks to run"""
    tasks = []
    
    # Get models to run
    if args.model:
        models = [args.model]
        # For benchmark version: Allow any model, don't require mapping
        if args.model not in mapping:
            print(f"Note: Model {args.model} not in mapping - allowing any test")
    elif args.family:
        models = get_models_from_family(args.family, mapping)
        if not models:
            return []
    
    # Determine which tests to run for each model
    for model in models:
        if args.test:
            specified_tests = args.test.split(',')
            if model in mapping:
                available_tests = set(mapping[model])
                for test in specified_tests:
                    if test in available_tests or test.endswith('_temp'):
                        tasks.append((model, test))
                    else:
                        print(f"Warning: Model {model} does not have test {test} configured, allowing anyway")
                        tasks.append((model, test))  # Allow anyway for benchmark
            else:
                # No mapping constraints - allow all tests
                tasks.extend([(model, test) for test in specified_tests])
        else:
            # Run all mapped tests for the model except those starting with "temp_"
            if model in mapping:
                tasks.extend([(model, test) for test in mapping[model] if not test.startswith("temp_")])
            else:
                print(f"No mapping found for {model}, skipping auto-test selection")
    
    return tasks 

def run_single_test(model_name, test_name, params):
    """Run a single test with parameter combination"""
    # Set current model context for logging
    custom_print.current_model = model_name
    
    print(f"\n{'-'*50}")
    print(f"Running test: {test_name}")
    # print(f"Parameters: {json.dumps(params, indent=2)}")
    test_start_time = time.time()
    
    # Build output path
    output_path = os.path.join(OUTPUT_ROOT, test_name, model_name)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Build command line arguments
    cmd_args = ['python', 'core/pi_flow_upgrade.py']
    cmd_args.extend(params_to_cmd_args(params))
    cmd_args.extend(['--output_path', output_path])
    
    print(f"Command: {' '.join(cmd_args)}")
    print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run test
    try:
        # Set encoding to handle UTF-8 output properly
        process_result = subprocess.run(
            cmd_args, 
            check=True, 
            capture_output=True, 
            text=True, 
            encoding='utf-8', 
            errors='replace'
        )
        
        # Log subprocess output
        if process_result.stdout:
            # Split and print each line for better readability
            print("Process output:")
            for line in process_result.stdout.splitlines():
                print(f"    | {line}")
        if process_result.stderr:
            print("Process error:")
            for line in process_result.stderr.splitlines():
                print(f"    | {line}")
        success = True
        print(f"Test completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Test failed: {e}")
        if e.stdout:
            print("Process output before failure:")
            for line in e.stdout.splitlines():
                print(f"    | {line}")
        if e.stderr:
            print("Process error output:")
            for line in e.stderr.splitlines():
                print(f"    | {line}")
        success = False
        print(f"Test failed with exit code {e.returncode}")
    
    test_duration = time.time() - test_start_time
    print(f"Test duration: {test_duration:.2f} seconds")
    print(f"Completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save the log after each test run
    log_path = save_log_buffer(model_name, test_name)
    print(f"Log saved to: {log_path}")
    print(f"{'-'*50}")
    
    # Clear model context
    custom_print.current_model = None
    
    return success

def run_test(model_name, test_name, continue_from_existing=False):
    """Run all parameter combination tests for specified model"""
    # Set current model context for logging
    custom_print.current_model = model_name
    
    print(f"\n{'='*50}")
    print(f"Starting test configuration: {test_name}")
    print(f"{'='*50}")
    start_time = time.time()
    
    # Load test configuration
    config = load_test_config(test_name)
    params = config['parameters']
    
    # Set model name
    params['model_name'] = model_name
    
    # Find existing parameter combinations if continuing
    existing_combinations = None
    if continue_from_existing:
        existing_combinations = find_existing_parameter_combinations(test_name, model_name)
        print(f"Found {len(existing_combinations)} existing parameter combinations")
        if existing_combinations:
            print("Existing combinations:")
            for combo in existing_combinations:
                print(f"  {combo}")
    
    # Generate parameter combinations
    param_combinations = generate_parameter_combinations(params, existing_combinations)
    
    print(f"Found {len(param_combinations)} parameter combinations to run")
    print(f"Output directory: {os.path.join(OUTPUT_ROOT, test_name, model_name)}")
    
    # Run all parameter combinations
    results = []
    for i, param_combo in enumerate(param_combinations, 1):
        print(f"\nRunning combination {i}/{len(param_combinations)}")
        success = run_single_test(model_name, test_name, param_combo)
        results.append(success)

    
    
    # Print test results
    duration = time.time() - start_time
    success_count = sum(results)
    print(f"\nTest {test_name} completed:")
    print(f"Total combinations: {len(param_combinations)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(param_combinations) - success_count}")
    print(f"Time taken: {duration:.2f} seconds")
    print(f"Results saved in: {os.path.join(OUTPUT_ROOT, test_name, model_name)}")
    
    # Clear model context
    custom_print.current_model = None
    
    return all(results)

def main():
    args = parse_args()
    mapping = load_mapping()
    tasks = get_tasks_to_run(mapping, args)
    
    if not tasks:
        print("No tests found to run!")
        return
    
    # Initialize log buffers for all models that will be tested
    for model, _ in tasks:
        get_or_create_log_buffer(model)
    
    # Set no specific model for global logs
    custom_print.current_model = None
    
    print(f"\n{'='*50}")
    print(f"TEST RUN CONFIGURATION")
    print(f"{'='*50}")
    print(f"Will run the following tests:")
    model_count = len(set(model for model, _ in tasks))
    test_count = len(set(test for _, test in tasks))
    for model, test in tasks:
        print(f"- Model: {model}, Test: {test}")
    if args.family:
        print(f"Model Family: {args.family} ({model_count} models)")
    else:
        print(f"Model: {args.model}")
    print(f"Total: {model_count} model(s), {test_count} test type(s), {len(tasks)} total tasks")
    print(f"Output root directory: {OUTPUT_ROOT}")
    if args.resume:
        print("Resuming from existing results, skipping completed parameter combinations")
    print("=" * 50)
    
    # Run all tests
    total_start_time = time.time()
    results = []
    
    for model, test in tasks:
        success = run_test(model, test, args.resume)
        results.append((model, test, success))
    
    # Print summary report
    total_time = time.time() - total_start_time
    print(f"\n{'='*50}")
    print(f"SUMMARY REPORT")
    print(f"{'='*50}")
    for model, test, success in results:
        status = "Success" if success else "Failed"
        print(f"Model {model}, Test {test}: {status}")
        print(f"Result directory: {os.path.join(OUTPUT_ROOT, test, model)}")
    print(f"\nTotal time taken: {total_time:.2f} seconds")
    
    # Save all model-specific logs
    for model, test, _ in results:
        log_path = save_log_buffer(model, test)
        print(f"Terminal log for model {model} saved to: {log_path}")

# Restore original print function at exit
import atexit
def restore_print():
    global print
    print = original_print
atexit.register(restore_print)

if __name__ == "__main__":
    # Initialize the log filename for this run
    get_log_filename()
    
    # Log start information will be done for each model in get_or_create_log_buffer
    main() 