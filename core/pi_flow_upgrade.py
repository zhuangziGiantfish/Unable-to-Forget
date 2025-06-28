import os
import sys
import json
import random
import re
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import time
import warnings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from fractions import Fraction
import shutil

# Add benchmark root directory to Python path for module imports
benchmark_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if benchmark_root not in sys.path:
    sys.path.insert(0, benchmark_root)

# Import necessary components from core/chat_terminal.py
from core.chat_terminal import get_client, ChatHistoryManager, TokenCounter, load_api_keys, call_api, judge_model_family, try_use_snapshot_model

# 定义一个自定义异常类用于配额限制
class QuotaExceededError(Exception):
    """Exception raised when API quota is exceeded."""
    pass

## setting the retry to 1: no retry
@retry(
    retry=retry_if_exception_type((QuotaExceededError, ValueError)),
    stop=stop_after_attempt(1),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    before_sleep=lambda retry_state: print(f"API call failed. Retrying in {retry_state.next_action.sleep} seconds... (Attempt {retry_state.attempt_number}/5)")
)
def call_api_with_retry(client, model_name, model_family, history, max_tokens, temperature):
    """
    Call API with automatic retry for quota errors or when response is None
    """
    response = call_api(client, model_name, model_family, history, max_tokens, temperature)
    
    # Retry if response is None
    if response is None:
        print("Received None response from API call")
        raise ValueError("API returned None response")
    
    ## the word error in the response (case insensitive)
    elif ("error" in response.lower()) and (len(response) < 200):
        raise ValueError("API returned error response")
        
    # Handle quota errors
    try:
        return response
    except Exception as e:
        error_msg = str(e)
        print(f"\nPrinted error_msg variable: {error_msg}\n")
        if ("429" in error_msg and "Resource has been exhausted" in error_msg and 
            model_family == "gemini"):
            raise QuotaExceededError("Gemini API quota exceeded") from e
        # 可以添加其他API的特定错误处理
        elif "rate_limit" in error_msg.lower() or "rate-limit" in error_msg.lower() or "429" in error_msg:
            raise QuotaExceededError(f"API rate limit exceeded: {error_msg}") from e
        else:
            # 重新抛出其他错误
            raise


def initialize_test(args):
    """
    Initialize the test environment
    
    Args:
        args: Command line arguments
        
    Returns:
        client: LLM client
        chatman: Chat history manager
        token_counter: Token counter
        test_name: Test name
        output_path: Output path
    """    
    # Generate test name
    date_time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    test_name = f"{date_time_str}_{args.test_nick_name}"

    # Create output directory
    output_path = Path(args.output_path) / test_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Load API keys
    api_keys = load_api_keys()

    model_general_name = args.model_name 
    model_snapshot_name = try_use_snapshot_model(model_general_name,"configs/models_snapshots.json")

    # Initialize LLM client
    client = get_client(model_snapshot_name, api_keys, args.temperature, args.max_tokens)
    
    # Initialize ChatHistoryManager
    chatman = ChatHistoryManager(model_general_name, output_path, args.test_nick_name)
    
    # Initialize token counter
    model_family = judge_model_family(model_general_name)
    token_counter = TokenCounter(model_general_name, client if model_family == "gemini" else None)
            
    # Create metafile
    metafile_path = output_path / f"{test_name}_meta-info.json"

    ## modify args by adding date_time and finished_sessions
    args.date_time = date_time_str
    args.finished_sessions = 0
    meta_info = vars(args)
    with open(metafile_path, 'w') as f:
        json.dump(meta_info, f, indent=4)
    
    print(f"Test initialized: {test_name}")
    print(f"Output directory: {output_path}")
    
    return client, chatman, token_counter, test_name, output_path, model_snapshot_name


def load_source_dictionary(source_dict_path):
    """
    Load the source dictionary from a JSON file
    
    Args:
        source_dict_path: Path to the source dictionary JSON file
        
    Returns:
        source_dict: Source dictionary
    """
    with open(source_dict_path, 'r') as f:
        source_dict = json.load(f)
    return source_dict

def track_usage(keys, items, used_keys, used_items):
    """
    Track the usage of keys and items to ensure balanced selection
    
    Args:
        keys: List of all keys
        items: Dictionary mapping keys to items
        used_keys: Dictionary tracking key usage count
        used_items: Dictionary tracking item usage count
        
    Returns:
        used_keys: Updated key usage tracker
        used_items: Updated item usage tracker
    """
    # Initialize tracking dictionaries if they don't exist
    if not used_keys:
        used_keys = {key: 0 for key in keys}
    
    if not used_items:
        used_items = {}
        for key, key_items in items.items():
            for item in key_items:
                used_items[f"{key}:{item}"] = 0
    
    return used_keys, used_items

def select_keys_and_items(source_dict, n_keys, n_updates, used_keys=None, 
                             used_items=None, balanced_sample=False, sample_replacement=False):
    """
    Select keys and items based on usage statistics
    
    Args:
        source_dict: Source dictionary with keys and items
        n_keys: Number of keys to select
        n_updates: Number of items to select per key
        used_keys: Dictionary tracking key usage
        used_items: Dictionary tracking item usage
        
    Returns:
        list_used_keys: List of selected keys
        list_all_pairs: List of all key-item pairs
        used_keys: Updated key usage tracker
        used_items: Updated item usage tracker
    """
    keys = list(source_dict.keys())
    
    # Initialize or update usage tracking
    used_keys, used_items = track_usage(keys, source_dict, used_keys, used_items)
    
    # Convert usage counts to selection weights (inverse relationship)
    key_weights = [1.0 / (used_keys[key] + 1) for key in keys]
    
    # Select keys based on weights WITHOUT replacement
    num_keys_to_select = min(n_keys, len(keys))
    normalized_key_weights = np.array(key_weights) / sum(key_weights)
    if not balanced_sample:
        normalized_key_weights = None

    list_used_keys = np.random.choice(
        keys, 
        size=num_keys_to_select,
        replace=False,  # No duplicates
        p=normalized_key_weights
    ).tolist()
    
    # Update key usage count
    for key in list_used_keys:
        used_keys[key] += 1
    
    # Select items for each key
    list_all_pairs = []
    for key in list_used_keys:
        available_items = source_dict[key]
        # Calculate item weights based on usage
        item_weights = []
        for item in available_items:
            item_key = f"{key}:{item}"
            count = used_items.get(item_key, 0) + 1
            item_weights.append(1.0 / count)
        
        # Select items based on weights WITHOUT replacement using numpy
        # num_to_select = min(n_updates, len(available_items))
        num_to_select = n_updates

        ## warning if sample_replacement is 0, but n_updates is greater than the number of items in the key
        if (not sample_replacement) and (n_updates > len(available_items)):
            warnings.warn(f"n_updates is greater than the number of items in the key {key}, this may cause unintended behavior")
        
        # Normalize weights to sum to 1.0 as required by np.random.choice
        normalized_weights = np.array(item_weights) / sum(item_weights)

        if not balanced_sample:
            normalized_weights = None
        
        selected_indices = np.random.choice(
            range(len(available_items)),
            size=num_to_select,
            replace=sample_replacement,  # No duplicates (without replacement)
            p=normalized_weights
        )

        selected_items = [available_items[i] for i in selected_indices]
        
        # Update item usage count and add to list_all_pairs
        for item in selected_items:
            item_key = f"{key}:{item}"
            used_items[item_key] = used_items.get(item_key, 0) + 1
            list_all_pairs.append((key, item))
    
    return list_used_keys, list_all_pairs, used_keys, used_items


def pseudo_randomize(list_all_pairs, max_key_repeat=0):
    """
    Randomize a list of key-item pairs with control over consecutive key repetitions.
    
    Args:
        list_all_pairs: List of tuples where the first element is the key
        max_key_repeat: Maximum allowed consecutive repetitions of the same key
                        0 means no consecutive repetitions allowed
                        1 means the same key can appear at most twice in a row
                        
    Returns:
        A randomized list with the same pairs but satisfying the constraint
    """
    if not list_all_pairs:
        return []
    
    # Make a copy to avoid modifying the original list
    remaining_pairs = list_all_pairs.copy()
    # Shuffle the copy to start with a random ordering
    random.shuffle(remaining_pairs)
    
    result = [remaining_pairs.pop()]  # Start with a random pair
    
    while remaining_pairs:
        # Count the current consecutive repetitions of the last key
        repeat_count = 1
        for i in range(len(result)-1, -1, -1):
            if result[i][0] == result[-1][0]:
                repeat_count += 1
            else:
                break
                
        # If we've reached the maximum allowed repetitions, find a different key
        if repeat_count > max_key_repeat:
            # Find candidates with a different key than the last one
            candidates = [i for i, pair in enumerate(remaining_pairs) if pair[0] != result[-1][0]]
            
            if not candidates:
                # If no suitable candidates, we need to retry with a different arrangement
                # This is a simple approach; more sophisticated backtracking could be implemented
                return pseudo_randomize(list_all_pairs, max_key_repeat)
            
            # Select a random candidate
            next_idx = random.choice(candidates)
        else:
            # Otherwise, select any random pair
            next_idx = random.randrange(len(remaining_pairs))
        
        # Add the selected pair to the result and remove it from remaining pairs
        result.append(remaining_pairs.pop(next_idx))
    
    return result



def remix_category_dict(dict_original):
    """
    This function evenly picks items from all categories and returns a new dictionary with the items scrambled.
    """

    n_category = len(dict_original)

    list_item_first = list(dict_original.values())[0] 
    n_items = len(list_item_first) ## assume all the lists have the same length

    ## assert all the lists have the same length
    for k,list_v in dict_original.items():
        assert len(list_v) == n_items, f"In the original cateogory dictionary, {k} has {len(list_v)} items, expected {n_items}"

    n_more = n_items % n_category
    n_less = n_category - n_more

    size_less = n_items // n_category
    size_more = size_less + 1

    list_size = [size_less]*n_less + [size_more]*n_more
    vec_size = np.array(list_size)

    list_full = []
    ### roll the list by 1
    for i in range(n_category):
        vec_size = np.roll(vec_size, 1)
        list_full.append(vec_size.tolist())

    array_full = np.array(list_full)


    ## sample from dict_original without replacement (removing the used words) using array_full as the numbers
    ## combine the samples corresponding to the same row as a new list

    dict_original_copy = dict_original.copy()

    i = 0

    list_resample = [[] for _ in range(n_category)]

    for k,list_v in dict_original_copy.items():
        np.random.shuffle(list_v)

        ##divide list_v into n_category parts,the sizes correspond to the numbers in the ith row of array_full
        sec_sizes = array_full[i]
        sec_ids = sec_sizes.cumsum()
        sec_ids = sec_ids[:-1]
        
        parts = np.array_split(list_v, sec_ids)

        ## distribute the parts to the list_sample
        for j in range(n_category):
            list_resample[j] = list_resample[j] + parts[j].tolist()

        i+=1

    ## randomly assign the list in list_resample to the dict_sample
    dict_resample = {}
    for j in range(n_category):
        np.random.shuffle(list_resample[j])
        dict_resample[list(dict_original.keys())[j]] = list_resample[j]

    ## assert the length of the new lists are n_items
    for k,list_v in dict_resample.items():
        assert len(list_v) == n_items, f"In the resampled cateogory dictionary, {k} has {len(list_v)} items, expected {n_items}"

    print("the fully scrambled new cateogory dictionary has been successfully created")

    return dict_resample


            
def convert_a_string_to_number(str_number):
    """
    Convert a string representation of a number to a float.
    try int, float, Fraction
    if all fails, return np.nan
    """
    try:
        return int(str_number)
    except ValueError:
        try:
            return float(str_number)
        except ValueError:
            try:
                return float(Fraction(str_number))
            except ValueError:
                return np.nan
            

def convert_a_string_to_forget_at_list(str_forget_at,n_updates):
    """
    Convert a string representation of a forget at number to a float.
    try fraction, float, int
    if all fails, return np.nan
    """
    num = convert_a_string_to_number(str_forget_at)
    if num is np.nan:
        list_forget_at = {
            "each": list(range(1,n_updates)), ## without the first udpate which is the initial value, at each update
            "even": [i for i in range(n_updates) if i % 2 == 0],
            "odd": [i for i in range(n_updates) if i % 2 == 1],
        }[str_forget_at]
    elif 0 < num < 1:
        list_forget_at = [int(np.ceil(num * n_updates))]
    elif isinstance(num, int):
        if num > n_updates:
            list_forget_at = []
            warnings.warn(f"The forget at number {num} is greater than the number of updates {n_updates}, it will be ignored")
        elif 0 <= num < n_updates:
            list_forget_at = [num]
        elif num < 0:
            ## trace the forget at in reverse order, based on the n_updates
            list_forget_at = [n_updates + num]
    else:
        raise ValueError(f"The str_forget_at {str_forget_at} is not valid")

    return list_forget_at
        

def sample_keys(source_dict, n_keys, key_usage, balanced_sample=False):
    """
    Sample keys from the source dictionary based on usage history
    
    Args:
        source_dict: Source dictionary with keys and items
        n_keys: Number of keys to sample
        key_usage: Dictionary tracking key usage
        balanced_sample: Whether to balance sampling based on usage history
        
    Returns:
        list_keys: List of sampled keys
        key_usage: Updated key usage tracker
    """
    keys = list(source_dict.keys())

    if not keys:
        return [], key_usage
    
    # Initialize key usage if needed
    if not key_usage:
        key_usage = {key: 0 for key in keys}
    
    # Initialize counts for new keys that might have been added
    for key in keys:
        if key not in key_usage:
            key_usage[key] = 0
    
    # Calculate weights for sampling (inverse to usage frequency)
    if balanced_sample:
        key_weights = [1.0 / (key_usage[key] + 1) for key in keys]
        normalized_weights = np.array(key_weights) / sum(key_weights)
    else:
        normalized_weights = None
    
    # Select keys based on weights WITHOUT replacement
    num_keys_to_select = min(n_keys, len(keys))
    list_keys = np.random.choice(
        keys, 
        size=num_keys_to_select,
        replace=False,  # No duplicates
        p=normalized_weights
    ).tolist()
    
    # Update key usage count
    for key in list_keys:
        key_usage[key] += 1
    
    return list_keys, key_usage

def concat_items(items, concat_style):
    """
    Concatenate multiple items based on the specified style
    
    Args:
        items: List of items to concatenate
        concat_style: Style of concatenation
        
    Returns:
        concatenated_item: String of concatenated items
    """
    if concat_style == 'none':
        # Direct concatenation
        return ''.join(items)
    
    processed_items = []
    for item in items:
        if 'strip' in concat_style:
            # Remove spaces
            item = item.replace(' ', '')
        
        if 'cap' in concat_style:
            # Capitalize
            item = item[0].upper() + item[1:] if item else item
            
        processed_items.append(item)
    
    if 'hyphen' in concat_style:
        return '-'.join(processed_items)
    elif 'underscore' in concat_style:
        return '_'.join(processed_items)
    else:
        return ''.join(processed_items)
    

def sample_items(source_dict, list_keys, n_updates, item_usage, balanced_sample=False, sample_replacement=False, lengthen_item="1_none"):
    """
    Sample items for each key in list_keys
    
    Args:
        source_dict: Source dictionary with keys and items
        list_keys: List of keys to sample items for
        n_updates: Number of updates per key
        item_usage: Dictionary tracking item usage
        balanced_sample: Whether to balance sampling based on usage history
        sample_replacement: Whether to sample with replacement
        lengthen_item: String in format "rounds_style" for multi-round sampling and concatenation
        
    Returns:
        list_kv_pairs: List of key-value pairs
        item_usage: Updated item usage tracker
    """
    # Parse lengthen_item parameter
    if '_' in lengthen_item:
        try:
            rounds_str, concat_style = lengthen_item.split('_', 1)
            n_rounds = int(rounds_str)
        except ValueError:
            raise ValueError(f"Invalid lengthen_item parameter: {lengthen_item}. Format should be 'rounds_style'.")
    else:
        try:
            n_rounds = int(lengthen_item)
            concat_style = 'none'
        except ValueError:
            raise ValueError(f"Invalid lengthen_item parameter: {lengthen_item}. Should be an integer or 'rounds_style'.")
    
    # Validate concat_style
    valid_styles = {'none', 'strip', 'cap' ,'cap-strip', 'hyphen-strip', 'underscore-strip', 'hyphen-cap-strip', 'underscore-cap-strip'}
    if concat_style not in valid_styles:
        raise ValueError(f"Invalid concat_style: {concat_style}. Must be one of {valid_styles}.")
    
    # Initialize item usage if needed
    if not item_usage:
        item_usage = {}
        for key in source_dict.keys():
            for item in source_dict[key]:
                item_usage[f"{key}:{item}"] = 0
    
    list_kv_pairs = []
    
    # Dictionary to hold sampled items for each key for all rounds
    sampled_items_by_key = {key: [[] for _ in range(n_updates)] for key in list_keys}
    
    # Sample items for each round
    for round_idx in range(n_rounds):
        # Select items for each key
        for key in list_keys:
            available_items = source_dict[key]
            
            # Calculate item weights based on usage
            item_weights = []
            for item in available_items:
                item_key = f"{key}:{item}"
                # Initialize if not present
                if item_key not in item_usage:
                    item_usage[item_key] = 0
                count = item_usage[item_key] + 1
                item_weights.append(1.0 / count)
            
            # Select items based on weights
            num_to_select = n_updates
            
            # Warning if trying to sample without replacement but n_updates > available items
            if (not sample_replacement) and (n_updates > len(available_items)):
                warnings.warn(f"n_updates is greater than the number of items in the key {key}, this may cause unintended behavior")
            
            # Normalize weights
            if balanced_sample:
                normalized_weights = np.array(item_weights) / sum(item_weights)
            else:
                normalized_weights = None
            
            selected_indices = np.random.choice(
                range(len(available_items)),
                size=num_to_select,
                replace=sample_replacement,
                p=normalized_weights
            )
            
            selected_items = [available_items[i] for i in selected_indices]
            
            # Update item usage count 
            for item in selected_items:
                item_key = f"{key}:{item}"
                item_usage[item_key] = item_usage.get(item_key, 0) + 1
            
            # Store selected items for this round
            for i, item in enumerate(selected_items):
                sampled_items_by_key[key][i].append(item)
    
    # Concatenate items across rounds and create key-value pairs
    for key in list_keys:
        for update_idx in range(n_updates):
            items_for_concat = sampled_items_by_key[key][update_idx]
            # Only concatenate if there are multiple items (rounds > 1)
            if n_rounds > 1:
                concatenated_item = concat_items(items_for_concat, concat_style)
            else:
                concatenated_item = items_for_concat[0]  # Use the single item as is
            
            list_kv_pairs.append((key, concatenated_item))
    
    return list_kv_pairs, item_usage


def get_fake_conversation(model_family,dict_tracked_key_value,instruction,question,probe_target="current"):
    """dict_tracked_key_value is a dictionary of the current values of the tracked keys
    Currently, recommend only using this functionality for true gpt and gemini models, whose faked responses are used here
    """
    current_response = ""
    for key, item in dict_tracked_key_value.items():
        current_response += f"The {probe_target} value of {key} is {item}. \n"
    ## remove the last \n
    current_response = current_response.rstrip()
        
    if model_family == "gemini":
        model_role_name = "model"
    else:
        model_role_name = "assistant"

    role_model_prefix = '\n{\n\"role\": \"' + model_role_name + '\",\n\"content\": \"Okay, Here are the current values of the specified keys:\n\n'
        
    connect = '\"\n}\n],\n[\n{\n\"role\": \"user\",\n\"content\": \"'
    insert_text = "\n\n" + question + '\"\n}, ' + role_model_prefix + current_response + connect + instruction + '\n'
    
    return insert_text 


def prepare_input_text(source_dict, n_tracked_keys, n_untracked_keys, n_tracked_updates, n_untracked_updates, 
                      random_update=True, sample_replacement=False, tracked_key_usage=None, untracked_key_usage=None,
                      tracked_item_usage=None, untracked_item_usage=None, balanced_sample=False,
                      probe_target="current", prompt_updating="colon", prompt_forgetting="none",
                      answer_format="verbal", extra_output_format="redundant",
                      lengthen_item="1_none", model_family="gpt"):
    """
    Prepare the input text for the LLM with different update frequencies for tracked and untracked keys
    
    Args:
        source_dict: a dictionary with keys as categories and values as lists of items
        n_tracked_keys: Number of keys to track and query about
        n_untracked_keys: Number of keys to include but not track
        n_tracked_updates: Number of updates for tracked keys
        n_untracked_updates: Number of updates for untracked keys
        random_update: Whether to randomly permute the pairs
        sample_replacement: Whether to sample items with regard to a key with replacement
        tracked_key_usage: Dictionary tracking tracked key usage
        untracked_key_usage: Dictionary tracking untracked key usage
        tracked_item_usage: Dictionary tracking tracked item usage
        untracked_item_usage: Dictionary tracking untracked item usage
        balanced_sample: Whether to balance the sampling across different sessions
        probe_target: Influence which update to probe and how it is probed
        prompt_updating: Format of the basic updating style
        prompt_forgetting: Format of forgetting prompts
        answer_format: Format of the answer (verbal or dict)
        extra_output_format: Additional output format requirements
        lengthen_item: String in format "rounds_style" for multi-round sampling and concatenation

    Returns:
        input_text: Assembled input text
        list_all_pairs: List of all key-item pairs
        list_tracked_keys: List of tracked keys
        list_untracked_keys: List of untracked keys
        tracked_key_usage: Updated tracked key usage tracker
        untracked_key_usage: Updated untracked key usage tracker
        tracked_item_usage: Updated tracked item usage tracker
        untracked_item_usage: Updated untracked item usage tracker
        list_forget_keys: List of keys to forget
    """
    # Check if there are enough keys in the source dictionary
    total_keys_needed = n_tracked_keys + n_untracked_keys
    if total_keys_needed > len(source_dict):
        raise ValueError(f"Not enough keys in source_dict: need {total_keys_needed}, have {len(source_dict)}")
    
    # Sample tracked keys first
    list_tracked_keys, tracked_key_usage = sample_keys(
        source_dict, n_tracked_keys, tracked_key_usage, balanced_sample
    )
    
    # Sample untracked keys, ensuring no overlap with tracked keys
    untracked_source_dict = {k: v for k, v in source_dict.items() if k not in list_tracked_keys}
    list_untracked_keys, untracked_key_usage = sample_keys(
        untracked_source_dict, n_untracked_keys, untracked_key_usage, balanced_sample
    )
    
    # Sample items for tracked keys
    list_tracked_kv_pairs, tracked_item_usage = sample_items(
        source_dict, list_tracked_keys, n_tracked_updates, tracked_item_usage, 
        balanced_sample, sample_replacement, lengthen_item
    )
    
    # Sample items for untracked keys
    list_untracked_kv_pairs, untracked_item_usage = sample_items(
        source_dict, list_untracked_keys, n_untracked_updates, untracked_item_usage, 
        balanced_sample, sample_replacement, lengthen_item
    )
    
    # Combine all pairs according to the random_update parameter
    if random_update:
        list_all_pairs = list_tracked_kv_pairs + list_untracked_kv_pairs
        max_key_repeat = 0
        print(f"randomizing the list allowing {max_key_repeat} consecutive repetitions of the same key")
        list_all_pairs = pseudo_randomize(list_all_pairs, max_key_repeat=max_key_repeat)
    else:
        list_all_pairs = list_untracked_kv_pairs + list_tracked_kv_pairs
    
    # Determine which keys to forget (only from tracked keys)
    list_forget_setups = prompt_forgetting.split("_")
    assert 1 <= len(list_forget_setups) <= 4, f"Prompt forgetting {prompt_forgetting} not implemented"

    ## the mode can be "none", "forget", "fwdattend", "sayreset", "hackreset"
    list_forget_global_modes = ["sayreset", "hackreset", "fwdattend","mindful","meditation","imagine","mindfulnobreak"]
    str_forget_mode = list_forget_setups[0]

    ## how many keys to prompt forgetting
    try:
        str_portion_forget = list_forget_setups[1]
    except IndexError:
        str_portion_forget = "none" ## default forgetting half of the keys
    ## currently, make sure that for sayreset, hackreset, and fwdattend, str_portion_forget must be "none"
    if str_forget_mode in list_forget_global_modes and str_portion_forget != "none":
        raise ValueError(f"\n\nFor {str_forget_mode}, the second parameter of {prompt_forgetting} must be 'none' instead of {str_portion_forget}\n\n")

    ## the position of the update to prompt forgetting
    try:
        str_forget_at = list_forget_setups[2]
    except IndexError:
        str_forget_at = "each" ## default forgetting at
    
    try:
        str_forget_grammar = list_forget_setups[3]
    except IndexError:
        str_forget_grammar = "separate" ## default format

    ## number of keys to forget
    n_forget = {
        "none": 0,
        "half": int(n_tracked_keys/2),
        "full": n_tracked_keys,
    }[str_portion_forget]

    if str_forget_mode == "none":
        n_forget = 0
    elif str_forget_mode in ["fwdattend","sayreset", "hackreset","mindful","mindfulnobreak","imagine"]:
        ### "sayreset": special case where forgetting is not linked to a specific key
        ### "hackreset": special case where forgetting is not linked to a specific key
        n_forget = 1
    elif str_forget_mode == "activelocate":
        n_forget == 0
        streamloc_forget_at = None
    elif str_forget_mode == "meditation":
        n_forget == 0
        streamloc_forget_at = None
    elif str_forget_mode == "forget":
        pass
    else:
        raise ValueError (f"the forget model {str_forget_mode} is undefined")

    ## the keys to prompt forgetting is a subset of the tracked keys
    list_forget_keys = np.random.choice(list_tracked_keys, size=n_forget, replace=False).tolist()

    ## add another mode in which the forgetting is inserted at a fixed position withregard to the stream, not the key
    ## if str_forget_at is bracketed by [], then the forgetting is inserted at a fixed position with regard to the stream
    if str_forget_at.startswith("[") and str_forget_at.endswith("]"):
        ## this is a strong mode which will invalidate the key-related forgetting including "half" and "full" options
        ## remove the brackets
        str_forget_at = str_forget_at[1:-1]
        ## convert the string to a integer
        try:
            streamloc_forget_at = int(str_forget_at)
        except ValueError:
            raise ValueError(f"Invalid forgetting position {str_forget_at}. The valid form is [n], where n is an integer.")
        
        if streamloc_forget_at < 0:
            ## reverse the streamloc_forget_at
            streamloc_forget_at = streamloc_forget_at + len(list_all_pairs)
            
        ## We'll find the closest update position for each key at stream generation time
        ## Set list_forget_at to empty as it's not used for stream-based forgetting
        list_forget_at = []
    else:
        streamloc_forget_at = None
        list_forget_at = convert_a_string_to_forget_at_list(str_forget_at, max(n_tracked_updates, n_untracked_updates))

    # Format the updates with appropriate forgetting
    separate_updates_by = ";"
    str_format_update = prompt_updating.split("_")[0] ## leaves space for future extensions

    format_update = {
        "hyphen": lambda key,item: f"{key} - {item}{separate_updates_by} ",
        "colon": lambda key,item: f"{key}: {item}{separate_updates_by} ",
        "equal": lambda key,item: f"{key} = {item}{separate_updates_by} ",
        "wordy": lambda key,item: f"update the value of {key} by setting it to {item}{separate_updates_by} ",
    }[str_format_update]


    dict_count_key_update = {k: 0 for k in list_tracked_keys + list_untracked_keys}
    dict_tracked_key_value = {k: None for k in list_tracked_keys}
    stream_key_item_pairs = ""

    # If we're using stream-based forgetting with multiple keys, we need to track positions
    if streamloc_forget_at is not None and str_portion_forget in ["half", "full"]:
        # Dictionary to track the stream positions for each key's updates
        key_stream_positions = {k: [] for k in list_forget_keys}
        
        # First pass: record all positions where each key appears in the stream
        q = 0
        for key, item in list_all_pairs:
            if key in list_forget_keys:
                key_stream_positions[key].append(q)
            q += 1
        
        # For each key, find the update position closest to streamloc_forget_at
        key_forget_positions = {}
        for key, positions in key_stream_positions.items():
            if positions:  # Only process if the key has updates
                # Find the position closest to streamloc_forget_at
                closest_pos = min(positions, key=lambda pos: abs(pos - streamloc_forget_at))
                key_forget_positions[key] = closest_pos
    
    ######################### Generate the instruction before the updating stream #########################
    str_actually_tracked_keys = ", ".join(list_tracked_keys)
    n_actually_tracked_keys = len(list_tracked_keys)

    instruction = f"As my secretary, I need you to carefully read a text stream where the values of multiple keys are being continuously updated."
    instruction += f"The {n_actually_tracked_keys} keys to track include {str_actually_tracked_keys}. I will ask you to identify the value of each key later."
    ######################### End of generating the instruction before the updating stream #########################
    
    ######################### Generate the question after the updating stream #########################
    ## mention the probe target angain here
    question = f"What are the {probe_target} value of each key ({str_actually_tracked_keys}) you are tracking? "

    if extra_output_format == "succinct":
        question += f"Format your response as: "
    elif extra_output_format.startswith("redundant"):
        question += f"End your response with: "
    else:
        raise ValueError(f"Extra output style {extra_output_format} not implemented")

    if answer_format == "verbal":
        question += f"'The {probe_target} value of <key> is <value>.'"
        f"Ensure that you report each key exactly once in this manner. "
    elif answer_format == "dict":
        question += f"'The {probe_target} " + "key-value pairs are: {<key>: <value>, ...}' (similar to a Python dictionary, but without quotation marks around strings)"
        question += f"Ensure that you report each key exactly once in the dictionary."
    else:
        raise ValueError(f"Answer format {answer_format} not implemented")
        question = None
    if extra_output_format == "redundantIntegrity":
        question += f"Provide the exact {probe_target} value string without modification or breaking it into pieces."

    # question += f"Ensure that you report each key exactly once and in a structured manner. "

    if extra_output_format == "succinct":
        question += "No extra text should be included in the response apart from what is requested."

    ######################### End of generating the question after the updating stream #########################

    ######################### Generate the updating stream #########################
    # Format for forgetting
    if str_forget_mode == "fwdattend":
            format_update_forget = lambda key: "pay close attention to the following updates. "
    elif str_forget_mode == "sayreset":
        format_update_forget = lambda key: "this text stream ends here and a new session of text stream begins next. "
    elif str_forget_mode == "mindful":
        format_update_forget = lambda key: "let's take a break and simply rest your attention on the breath for a moment. feel each inhalation and exhalation. if you realize your attention is wandering, gently bring it back to the breath. "
    elif str_forget_mode == "mindfulnobreak":
        format_update_forget = lambda key: "now, simply rest your attention on the breath for a moment. feel each inhalation and exhalation. if you realize your attention is wandering, gently bring it back to the breath. "
    elif str_forget_mode == "imagine":
        format_update_forget = lambda key: "Picture a clear sky above you. Any tension or distraction is like a passing cloud—acknowledge it, then let it drift away into the open blue."
    elif str_forget_mode == "hackreset":
        format_update_forget = lambda key: get_fake_conversation(model_family,dict_tracked_key_value,instruction,question,probe_target)
    elif str_forget_mode == "forget":
        format_update_forget = {
            "separate": lambda key: f"forget all the previous updates to {key}{separate_updates_by} ",
            "inline": lambda key: f"forget all the previous updates to {key} then update it to ",
            "separate-control": lambda key: f"the next key to update is {key}{separate_updates_by} ",
            "inline-control": lambda key: f"let's update the value of {key} by setting it to ",
        }[str_forget_grammar]
    elif str_forget_mode == "activelocate":
        question += " Before answer it, analyze for this task, which portion of the text should be emphasized? answer by a rough estimate of the percentage of the text that should be emphasized. "
    elif str_forget_mode == "meditation":
        question += " Before answer it, let's do a 10-min session of focused-attention meditation. Rest your attention on your breath and feel each inhalation and exhalation. If you realize your attention is wandering, gently bring it back to the breath. "

    q = 0
    for key, item in list_all_pairs:
        # For stream-based forgetting with multiple keys
        if streamloc_forget_at is not None and str_portion_forget in ["half", "full"]:
            if key in list_forget_keys and q == key_forget_positions.get(key, -1):
                stream_key_item_pairs += format_update_forget(key)
        # Single position stream-based forgetting
        elif streamloc_forget_at is not None:
            if q == streamloc_forget_at:
                stream_key_item_pairs += format_update_forget(key)
        # Key-based forgetting
        elif key in list_forget_keys and dict_count_key_update[key] in list_forget_at:
            stream_key_item_pairs += format_update_forget(key)

        stream_key_item_pairs += format_update(key, item)
        dict_count_key_update[key] += 1
        if key in list_tracked_keys:
            dict_tracked_key_value[key] = item
        q += 1
    text_key_item_pairs = f'The text stream starts on the next line.\n {stream_key_item_pairs}'
    ######################## End of generating the updating stream ########################
    
    
    ######################### Generate the input text by combining the instruction, the updating stream, and the question #########################
    input_text = f"{instruction}\n\n{text_key_item_pairs}\n\n{question}"

    return (input_text, list_all_pairs, list_tracked_keys, list_untracked_keys, 
            tracked_key_usage, untracked_key_usage, tracked_item_usage, untracked_item_usage, list_forget_keys, dict_tracked_key_value)


def extract_response_dict(model_output):
    """
    The first version of extract_response_dict
    The second version is called extract_pieces_response_to_dict

    Extract the dictionary of key-value pairs from the model output
    
    Args:
        model_output: Output from the model
        
    Returns:
        dict_response: Dictionary of key-value pairs extracted from the output
    """
    if (not model_output) or (len(model_output)<10):
        return None
    
    if (len(model_output) < 150) and ("error" in model_output.lower()):
        ## the model did not return a response
        return None

    # Define patterns to try - only keeping patterns that require "queried key-value pairs"
    patterns = [
        # Pattern 1: Standard format with "The queried key-value pairs are: {key:value, ...}"
        (r"[Tt]he (?:most recent|final|last|latest|current|up-to-date|current|asked|queried|specified) key-value pairs are:?\s*{([^}]*)}", lambda m: m.group(1), ','),
        
        # Pattern 2: Format without brackets but with "The queried key-value pairs are: key:value, ..."
        (r"[Tt]he (?:most recent|final|last|latest|current|up-to-date|current|asked|queried|specified) key-value pairs are:?(.+?)(?:\n|$)", lambda m: m.group(1).strip(), ','),
    ]
    
    # Try each pattern in order
    for pattern, extract_func, separator in patterns:
        matches = re.findall(pattern, model_output, re.DOTALL)
        
        if matches:
            # Use the last match for patterns
            match_str = matches[-1]
            dict_response = {}
            
            # Handle both bracket-enclosed and non-bracket formats
            if '{' in match_str and '}' in match_str:
                # Extract content between brackets if present
                bracket_content = re.search(r'{([^}]*)}', match_str)
                if bracket_content:
                    match_str = bracket_content.group(1)
            
            pairs = match_str.split(separator)
            
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    dict_response[key.strip()] = value.strip()
                elif '-' in pair:
                    key, value = pair.split('-', 1)
                    dict_response[key.strip()] = value.strip()
            
            if dict_response:
                return dict_response
    
    ## by default return empty dict instead of None
    return {}


##################################################################################
##### The extract functions in use above copied from analysis_helper.py     
##### remember to update the analysis_helper.py or here after any changes   
##################################################################################

def extract_pieces_response_to_dict(model_output, probe_target="latest"):
    """
    Extract the dictionary of key-value pairs from the model output.
    First extract using verbal language match, then using colon match.
    Merge the two dictionaries, prioritizing keys from the verbal match.
    """
    ## for response with zero length, return None
    if len(model_output) == 0:
        return None
    
    ## check if "Error code" in the model output
    if "error code" in model_output.lower():
        return None
    
    ## check if start with error or Error
    if model_output.startswith("error") or model_output.startswith("Error"):
        return None

    ## check if "error" as a whole word in the model output (part of a word does not count)
    ## error can be prefixed or suffixed by a space, comma, period, etc.
    if (re.search(r'\berror\b', model_output, re.IGNORECASE)) and (len(model_output) < 680):
        ## 680 is about token length of 160
        ## the model did not return a response
        return None

    ## remove all backslashes
    model_output = remove_slash(model_output)

    model_output = re.sub(r'\*', '', model_output)

    dict_verbal_match = _extract_verbal_matches(model_output, probe_target)

    dict_colon_match = _extract_colon_matches(model_output)

    dict_merged = dict_colon_match.copy()
    dict_merged.update(dict_verbal_match)

    ## remove the key named "key" whether it is in dict_merged or not
    dict_merged.pop("key", None)

    return dict_merged


def remove_slash(text):
    """
    Remove all backslashes '\' from the text except those that are part of '\n'.
    """
    # Replace all occurrences of '\' that are not followed by 'n'
    # This pattern matches a backslash not followed by 'n'
    return re.sub(r'\\(?!n)', '', text)

def remove_post_ending_texts(text):
    """
    Check if the text contains any of the following signs: ',', '.', ';', ':', '\n'
    If so, remove anything after the first occurrence of these signs including the sign itself.
    """
    # Define the signs to look for
    signs = [',', '.', ';', ':', '\n']
    # Find the first occurrence of any sign
    min_index = len(text)
    for sign in signs:
        idx = text.find(sign)
        if idx != -1 and idx < min_index:
            min_index = idx
    if min_index < len(text):
        return text[:min_index]
    else:
        return text


def _extract_verbal_matches(model_output, probe_target="current"):
    """
    Extract the dictionary of key-value pairs from the model output
    Look for multiple matches of prefix_str 
    
    Args:
        model_output: Output from the model
        probe_target: The probe target to look for, terms like "up-to-date", "most recent", "latest", etc.

    Returns:
        dict_response: Dictionary of key-value pairs extracted from the output
    """
    # Define patterns to try
    if probe_target in ["last", "latest", "up-to-date", "final", "most recent", "current"]:
        patterns = [
            # Pattern 1: for phrases like "electronics: The last seen value is "Smarts peaker"," This is the least common pattern
            # Key appears before phrases like "the last/current/final value is..."
            r"([\"'\[\<]?\w+(?:\s+\w+)?[\"'\]\>]?)(?:\s*[:-]\s*)(?:the)?\s*(?:most recent|final|last seen|last|latest|current|up-to-date|asked|queried|specified)\s+(?:value|word|term)(?:s)?\s+(?:is|was)(?:\s*:\s*)?\s+([\"'\[\<]?\w+(?:\s+\w+)?[\"'\]\>]?)(?=\n|[,.;:]|$)",

            # Pattern 2: for phrases like "The value for 'dog' is 'poodle'" - without any qualifier
            # 'the' must appear
            r"(?:the)\s*(?:value|word|term)?(?:s)?(?:\s+\w+){0,1}\s+(?:with|for|of|to)?\s+(?:the )?(?:category|key)?\s*([\"'\[\<]?\w+(?:\s+\w+)?[\"'\]\>]?)\s+(?:is|was)(?:\s*:\s*)?\s+([\"'\[\<]?\w+(?:\s+\w+)?[\"'\]\>]?)(?=\n|[,.;:]|$)",

            # Pattern 3: for phrases like "The most recent value paired with the category 'dog' is 'poodle'" This is the most common pattern
            # 'the' is optional
            r"(?:the)?\s*(?:most recent|final|last|latest|current|up-to-date|asked|queried|specified)\s+(?:value|word|term)?(?:s)?(?:\s+\w+){0,1}\s+(?:with|for|of|to)?\s+(?:the )?(?:category|key)?\s*([\"'\[\<]?\w+(?:\s+\w+)?[\"'\]\>]?)\s+(?:is|was)(?:\s*:\s*)?\s+([\"'\[\<]?\w+(?:\s+\w+)?[\"'\]\>]?)(?=\n|[,.;:]|$)",
        ]
    else:
        warnings.warn(f"Probe target {probe_target} not implemented, return empty dict")
        return {}
    
    dict_response = {}
    
    # Try each pattern to find all matches
    for pattern in patterns:
        matches = re.findall(pattern, model_output, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            # Each match will be a tuple with (key, value)
            if len(match) >= 2:
                key, value = match[0], match[1]
                
                # Clean up the key and value
                key = re.sub(r'[\*\'"""''\[\]\{\}\(\)\<\>]', '', key).strip()
                value = re.sub(r'[\*\'"""''\[\]\{\}\(\)\<\>]', '', value).strip()

                key = remove_post_ending_texts(key)
                value = remove_post_ending_texts(value)
                
                # Add to dictionary if not empty
                if key and value:
                    ## if the same key appear multiple times, the last one will overwrite the previous ones
                    dict_response[key] = value

    return dict_response


def _extract_colon_matches(model_output):
    """
    Extract key-value pairs from model_output using colon-separated patterns.
    Strips '*', quotes, and spaces from keys and values.
    Only extracts if the model output contains specific patterns like "last value", "latest term", etc.
    """
    # Check if the model output contains any of the required patterns
    pattern_check = r"(?:most recent|final|last|latest|current|up-to-date|asked|queried)\s+(?:value|word|term)"
    if not re.search(pattern_check, model_output, re.IGNORECASE):
        return {}
        
    # Updated pattern to capture keys that might be in quotes, brackets, etc.
    # Modified to treat newlines as delimiters for values
    colon_pattern = r'(?:[-,.;:\n]|\n|^)\s*[\'"""''\[\]\<\>]*(\w+(?:[ \t]+\w+)?)[\'"""''\[\]\<\>]*\s*:\s*[\'"""''\[\]\<\>]*(\w+(?:[ \t]+\w+)?)(?=[\'"""''\[\]\<\>]*\s*(?:\n|[,.;:]|$))'
    colon_pairs = re.findall(colon_pattern, model_output)
    dict_colon = {}
    for ckey, cvalue in colon_pairs:
        # Strip *, quotes, and then spaces (head/tail)
        ckey = re.sub(r'[\*\'"""''\[\]\{\}\(\)\<\>]', '', ckey).strip()
        cvalue = re.sub(r'[\*\'"""''\[\]\{\}\(\)\<\>]', '', cvalue).strip()
        if ckey and cvalue:
            ## if the same key appear multiple times, the last one will overwrite the previous ones
            dict_colon[ckey] = cvalue
    return dict_colon

##################################################################################
#     The extract functions in use above copied from analysis_helper.py     ######
#     remember to update the analysis_helper.py or here after any changes   ######
##################################################################################


def save_to_json(data, filepath):
    """
    Save data to a JSON file
    
    Args:
        data: Data to save
        filepath: Path to the file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def compute_ci95_width_by_bootstrap(acc, n=5000, outlier_sd=0):
    """
    Compute the width of the 95% confidence interval by bootstrapping.
    Optionally exclude values outside mean ± outlier_sd * std from acc before bootstrapping.
    If after filtering, less than 50% of items remain, use all values instead.

    Args:
        acc (list or array): List of accuracy values.
        n (int): Number of bootstrap samples.
        outlier_sd (float): Number of standard deviations for outlier exclusion.

    Returns:
        tuple: (width of the 95% confidence interval, number of filtered out trials)
    """
    ## check if len(acc) > 1
    if len(acc) <= 1:
        return np.nan, 0

    acc = np.array(acc)
    original_len = len(acc)
    filtered_acc = acc
    n_filtered_out = 0
    if outlier_sd > 0 and original_len > 1:
        mean = np.mean(acc)
        std = np.std(acc)
        lower = mean - outlier_sd * std
        upper = mean + outlier_sd * std
        mask = (acc >= lower) & (acc <= upper)
        filtered_acc = acc[mask]
        n_filtered_out = original_len - len(filtered_acc)
        # If less than 50% remain, use all values
        if len(filtered_acc) < 0.5 * original_len:
            filtered_acc = acc
            n_filtered_out = 0

    acc = filtered_acc
    arr = [np.mean(np.random.choice(acc, len(acc), True)) for _ in range(n)]
    arr.sort()
    width = np.percentile(arr, 97.5) - np.percentile(arr, 2.5)
    return width, n_filtered_out


def compute_accuracy(dict_response, dict_tracked_key_value):
    """
    Compute accuracy by comparing dict_response and dict_tracked_key_value.
    no response at all will be counted as np.nan(missing)
    no valid response will be counted as 0 (wrong)

    Args:
        dict_response (dict): The dictionary extracted from model output.
        dict_tracked_key_value (dict): The ground truth dictionary of tracked keys and their values.

    Returns:
        accuracy (float): Proportion of tracked keys correctly answered.
        n_missing (int): Number of tracked keys not mentioned in dict_response.
    """
    ## if dict_response is None, return np.nan and len(dict_tracked_key_value)
    if (dict_response is None) or not isinstance(dict_response, dict):
        return np.nan, np.nan
    elif len(dict_response) == 0:
        return 0.0, len(dict_tracked_key_value)
    elif not dict_tracked_key_value or not isinstance(dict_tracked_key_value, dict):
        return np.nan, np.nan

    n_total = len(dict_tracked_key_value)
    n_correct = 0
    n_missing = 0

    for key, true_value in dict_tracked_key_value.items():
        if key not in dict_response:
            n_missing += 1
        else:
            # Compare as string, ignore case and strip whitespace
            pred_value = str(dict_response[key]).strip().lower()
            true_value_str = str(true_value).strip().lower()
            if pred_value == true_value_str:
                n_correct += 1

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    return accuracy, n_missing


def run_pi_test(args):
    """
    Run the projective interference test
    
    Args:
        args: Command line arguments
    """
    # Initialize test environment
    client, chatman, token_counter, test_name, output_path, model_snapshot_name = initialize_test(args)
    
    # Initialize trackers
    tracked_key_usage = {}
    untracked_key_usage = {}
    tracked_item_usage = {}
    untracked_item_usage = {}
    collection_list_all_pairs = []
    dict_runs_of_list_response = {}
    dict_runs_of_actual_tracked_answers = {}
    collection_list_tracked = []
    collection_list_untracked = []
    check_histories = []
    collection_list_forget_keys = []
    
    model_family = judge_model_family(model_snapshot_name)
    

    df_trial = pd.DataFrame(columns=["n_tracked_keys", "n_untracked_keys", "n_tracked_updates", "n_untracked_updates", "len_item", 
                                     "session", "n_input_tokens", "n_output_tokens", "accuracy", 
                                     "n_missing","width_ci95","width_ci95_n_removed","dur_api_call"])

    answer_format = args.response_format.split("_")[0]
    ## this will influence the input text and how answer is extracted
    try:
        extra_output_format = args.response_format.split("_")[1]
    except IndexError:
        extra_output_format = "succinct"

    ## load the source dictionary
    source_dict = load_source_dictionary(args.source_dict_path)
    if args.remix_category:
        source_dict = remix_category_dict(source_dict)
    n_keys_available = len(source_dict)

    ## resolve tracked, untracked, total key numbers
    ## one of n_total_keys and n_untracked_keys must be 0
    assert args.n_total_keys * args.n_untracked_keys == 0, "In the settings, one of n_total_keys and n_untracked_keys must be 0"

    ## reassign values
    if args.n_total_keys == 0:
        args.n_total_keys = args.n_tracked_keys + args.n_untracked_keys
    elif args.n_untracked_keys == 0:
        args.n_untracked_keys = args.n_total_keys - args.n_tracked_keys
        if args.n_untracked_keys < 0:
            raise ValueError(f"n_untracked_keys (=n_total_keys - n_tracked_keys) = {args.n_untracked_keys} is less than 0")
    assert args.n_total_keys <= n_keys_available, f"n_total_keys (= {args.n_total_keys}) is greater than the number of available keys (= {n_keys_available}) in {args.source_dict_path}"
    
    ## the argument lengthen_item has been separated into two arguments
    if args.lengthen_item == "deprecated":
        args.lengthen_item = f"{args.len_item}_{args.len_item_style}"

    ## implement ci95 width threshold and max sessions if not provided
    ## ci95 is computed by bootstrap when sessions > min_sessions
    min_sessions = 5
    ## set up mas number of sessions to 30 for deepseek-reasoner, the rest with 50
    if args.model_name == "deepseek-reasoner":
        max_sessions = 20
    elif args.model_name in ["gpt-4.1", "gpt-4", "gpt-4o"]:
        max_sessions = 20
    elif args.model_name.startswith("qwen"):
        max_sessions = 20
    elif args.model_name.startswith("claude"):
        max_sessions = 20
    else:
        max_sessions = 50

    if args.n_sessions == 0:
        n_session = max_sessions
        if args.model_name == "deepseek-reasoner":
            ci95_width_threshold = 0.15
        else:
            ci95_width_threshold = 0.1
        n_consecutive_sessions = 2
        print(f"No n_sessions provided, using a max of {n_session}, stop running when CI95 width < {ci95_width_threshold}")

    else:
        n_session = args.n_sessions
        print(f"n_sessions provided, using {n_session}, no ci95 and no early stopping")

    list_acc = []
    list_ci95 = []
    list_responses = []
    list_tracked_answers = []
    
    # Check for previous runs with the same parameters
    # Extract parameter part of the test name (removing date-time prefix)
    param_suffix = "_".join(test_name.split("_")[2:])  # Skip the date and time parts
    parent_dir = Path(args.output_path)
    
    # Find all directories in the parent folder
    all_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]
    matching_dirs = []
    
    ######################### Find previous runs with the same parameters #########################
    ######################### compute the accumulated accuracy and ci95 #########################
    for directory in all_dirs:
        dir_name = directory.name
        if dir_name != test_name and dir_name.endswith(param_suffix):
            matching_dirs.append(directory)
    
    if matching_dirs:
        print(f"\nFound {len(matching_dirs)} previous runs with the same parameters.")
        
        # Process each matching directory
        for prev_dir in matching_dirs:
            responses_file = prev_dir / f"{prev_dir.name}_responses.json"
            answers_file = prev_dir / f"{prev_dir.name}_correct-answers.json"
            
            if responses_file.exists() and answers_file.exists():
                try:
                    with open(responses_file, 'r') as f:
                        prev_responses = json.load(f)
                    
                    with open(answers_file, 'r') as f:
                        prev_answers = json.load(f)
                    
                    # Compute accuracies for each session in the previous run
                    for i in range(min(len(prev_responses), len(prev_answers))):
                        if prev_responses[i] is not None and prev_answers[i] is not None:
                            acc, _ = compute_accuracy(prev_responses[i], prev_answers[i])
                            if not np.isnan(acc):
                                list_acc.append(acc)
                    
                    print(f"Added {len(list_acc)} previous accuracy values from {prev_dir.name}")
                    
                except Exception as e:
                    print(f"Error processing previous run {prev_dir.name}: {e}")
        
        # Compute the latest two sessions of CI95 width
        if (len(list_acc) > min_sessions) and (args.n_sessions == 0):
            for ii in range(n_consecutive_sessions):
                # If ii == 0, use the whole list; otherwise, exclude the last ii elements
                if ii == 0:
                    acc_subset = list_acc[:]
                else:
                    acc_subset = list_acc[:-ii]
                width_ci95, _ = compute_ci95_width_by_bootstrap(acc_subset, n=10000, outlier_sd=0)
                list_ci95.append(width_ci95)
            
            ## check if all values in list_ci95 are less than ci95_width_threshold
            if all(w < ci95_width_threshold for w in list_ci95):
                print(f"\nSession skiped before starting: {n_consecutive_sessions} consecutive CI95 widths {list_ci95} are less than {ci95_width_threshold}")
                
                ## remove the output directory
                shutil.rmtree(output_path)
                print(f"Output directory removed: {output_path}")
                return
            
            print(f"Starting with {len(list_acc)} accumulated accuracy values, mean: {np.mean(list_acc):.4f}, CI95 width: {list_ci95[-1]:.4f}")
            

    ######################### Run the projective interference test #########################
    for session in range(n_session):

        t_start = time.time()

        print(f"\nRunning session {session+1}/{n_session} | {args.model_name} ")
        
        # Prepare input text
        input_texts, list_all_pairs, list_tracked_keys, list_untracked_keys, \
        tracked_key_usage, untracked_key_usage, tracked_item_usage, untracked_item_usage, list_forget_keys, dict_tracked_key_value = \
        prepare_input_text(
            source_dict, 
            args.n_tracked_keys, 
            args.n_untracked_keys, 
            args.n_tracked_updates, 
            args.n_untracked_updates, 
            args.random_update,
            args.sample_replacement,
            tracked_key_usage,
            untracked_key_usage,
            tracked_item_usage,
            untracked_item_usage,
            args.balanced_sample,
            args.probe_target,
            args.prompt_updating,
            args.prompt_forgetting,
            answer_format,
            extra_output_format,
            args.lengthen_item,
            model_family
        )
    
        # Save tracking statistics
        tracking_stats = {
            "tracked_keys": tracked_key_usage,
            "untracked_keys": untracked_key_usage
        }
        save_to_json(tracking_stats, output_path / f"{test_name}_tracking-stats.json")
        
        # Save item usage statistics
        item_usage_stats = {
            "tracked_items": tracked_item_usage,
            "untracked_items": untracked_item_usage
        }
        save_to_json(item_usage_stats, output_path / f"{test_name}_item-usage-stats.json")
        
        # Add pairs to collection and save
        collection_list_all_pairs.append(list_all_pairs)
        save_to_json(collection_list_all_pairs, output_path / f"{test_name}_all-pairs.json")
        
        # Add tracked keys to collection and save
        collection_list_tracked.append(list_tracked_keys)
        save_to_json(collection_list_tracked, output_path / f"{test_name}_tracked.json")

        # Add untracked keys to collection and save
        collection_list_untracked.append(list_untracked_keys)
        save_to_json(collection_list_untracked, output_path / f"{test_name}_untracked.json")

        # save collection_list_forget_keys to json
        collection_list_forget_keys.append(list_forget_keys)
        save_to_json(collection_list_forget_keys, output_path / f"{test_name}_forget.json")


        print(f"Session {session}, starts")

        # Clear chat history if needed
        if args.memory_limit >= 1 and session % args.memory_limit == 0 and session > 0:
            chatman.reset()
            print("Chat history cleared")
    
        # Add user message to chat history
        chatman.add_user_message(input_texts)
        
        # Count input tokens
        input_tokens = token_counter.add_input_tokens(input_texts)
        print(f"Input tokens: {input_tokens}")
        
        ############# Call LLM API with retry logic #############
        t_api_start = time.time()
        try:
            
            model_output = call_api_with_retry(client, model_snapshot_name, model_family, 
                                            chatman.history, args.max_tokens, args.temperature)
            
        except QuotaExceededError as e:
            print(f"Failed after multiple retries: {e}")
            model_output = f"ERROR: API quota exceeded after multiple retries."
        except Exception as e:
            print(f"Unexpected error: {e}")
            model_output = f"ERROR: {str(e)}"

        t_api_end = time.time()
        dur_api = round(t_api_end - t_api_start, 2)
        print(f"API call time: {dur_api} seconds")

        # Count output tokens
        output_tokens = token_counter.add_output_tokens(model_output)
        print(f"Output tokens: {output_tokens}")
        
        # Add model response to chat history
        chatman.add_model_message(model_output)

        # save the history of the first few sessions for eyeballing
        if session <= 3:
            check_histories.append(chatman.history)
            with open(output_path / f"{test_name}_session123-history-check.json", "w", encoding="utf-8") as f:
                json.dump(check_histories, f, indent=2, ensure_ascii=False)
        else:
            check_histories = []

        # Extract response dictionary, two different ways
        if answer_format == "dict":
            dict_response = extract_response_dict(model_output) ## this extract all tracked keys at onece
        elif answer_format == "verbal":
            dict_response = extract_pieces_response_to_dict(model_output, args.probe_target) ## this extract tracked keys one by one
        else:
            warnings.warn(f"Answer format ({answer_format}) from response format ({args.response_format}) not implemented, return None")
            dict_response = None

        # Add raw output to file
        with open(output_path / f"{test_name}_raw-output.json", 'a') as f:
            json.dump({f"session-{session+1}": model_output}, f)
            f.write("\n")
        
        # arrange the responses by runs. each run contains a list of responses from multiple sessions
        list_responses.append(dict_response)
        list_tracked_answers.append(dict_tracked_key_value)

        save_to_json(list_responses, output_path / f"{test_name}_responses.json")
        save_to_json(list_tracked_answers, output_path / f"{test_name}_correct-answers.json")

        ## compute the accuracy of the current run
        ## the accuracy is computed by comparing the dict_response with the dict_tracked_key_value
        accuracy, n_missing = compute_accuracy(dict_response, dict_tracked_key_value)

        list_acc.append(accuracy)
        list_acc_notna = [a for a in list_acc if not np.isnan(a)]

        total_sessions = len(list_acc)
        total_sessions_notnan = len(list_acc_notna)

        # The number of sessions from previous runs
        prev_sessions = total_sessions - (session + 1)
        
        if total_sessions_notnan > min_sessions:
            width_ci95, n_filtered_out = compute_ci95_width_by_bootstrap(list_acc_notna, n=10000, outlier_sd=0)
        else:
            width_ci95 = np.nan
            n_filtered_out = np.nan

        list_ci95.append(width_ci95)

        # append a row to df_trial and save it to the output folder
        ## if df_trial is empty, create a new one to avoid warning of appending to an empty dataframe
        df_trial_new = pd.DataFrame([[args.n_tracked_keys, args.n_untracked_keys, 
                                                    args.n_tracked_updates, args.n_untracked_updates, args.lengthen_item,
                                                    session, input_tokens, output_tokens, accuracy, n_missing, 
                                                    width_ci95, n_filtered_out,dur_api]], columns=df_trial.columns)
        if len(df_trial) == 0:
            df_trial = df_trial_new.copy()
        else:
            df_trial = pd.concat([df_trial, df_trial_new], ignore_index=True)

        df_trial.to_csv(output_path / f"{test_name}_trial.csv", index=True)

        ## update the meta_info by setting finished sessions to current session
        args.finished_sessions = f"{session+1}/{n_session}"
        if prev_sessions > 0:  # If we have accumulated sessions from previous runs
            args.accumulated_sessions = prev_sessions
            args.total_sessions = total_sessions
            args.total_sessions_notnan = total_sessions_notnan
        meta_info = vars(args)
        with open(output_path / f"{test_name}_meta-info.json", 'w') as f:
            json.dump(meta_info, f, indent=4)


        ## if the last four consecutive values in list_responses are all None, exit this session
        n_consecutive_none = 3
        if len(list_responses) >= n_consecutive_none:
            if all(v is None for v in list_responses[-n_consecutive_none:]):
                print(f"Up to session {session+1}, found {n_consecutive_none} consecutive None responses, exiting")
                
                ## raise an error to stop
                raise ValueError(f"Up to session {session+1}, has 4 consecutive None responses, exiting")

        print(f"\n{session+1} sessions have been completed")
        print(f"Accuracy: {accuracy}, missing keys: {n_missing}")
        print(f"Response dictionary: {dict_response}")
        print(f"Tracked keys: {list_tracked_keys}")
        print(f"untracked keys: {list_untracked_keys}\n")

        # If the total number of sessions is greater than min_sessions,
        # check the last two width_ci95 values in df_trial.
        # If both are less than ci95_width_threshold, break the loop.
        if (total_sessions_notnan > min_sessions) and (args.n_sessions == 0):
            last_n_widths = list_ci95[-n_consecutive_sessions:]
            if len(last_n_widths) == n_consecutive_sessions and all(w < ci95_width_threshold for w in last_n_widths):
                print(f"Early stopping: last {n_consecutive_sessions} CI95 widths ({last_n_widths}) < threshold ({ci95_width_threshold})")
                break

        # Add a small delay between sessions to avoid rate limiting
        if session < n_session - 1:
            time.sleep(2)

        # if model_snapshot_name.startswith("gemini-2.0-flash-thinking") or model_snapshot_name.startswith("gemini-2.5-pro"):
        #     time.sleep(5)

        ## extra sleep for claude as it limits the number of requests per minute to 20000 tokens
        if model_snapshot_name.startswith("claude"):
            time.sleep(15)

    print(f"\nTest completed: {test_name}")
    print(f"Results saved to: {output_path}")

    return

def main():
    parser = argparse.ArgumentParser(description="Run Projective Interference Test")
    
    # LLM parameters
    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash",
                        help="LLM model to use")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for LLM generation")
    parser.add_argument("--max_tokens", type=int, default=2000,
                        help="Maximum tokens for model output")
    
    # Test parameters
    parser.add_argument("--test_nick_name", type=str, default="pi_test",
                        help="Nickname for the test")
    parser.add_argument("--output_path", type=str, default="eval_pi_temp",
                        help="Path to output directory")
    parser.add_argument("--source_dict_path", type=str, default="testing_data/dict_category_double-word_46-400_v1-1.json",
                        help="Path to source dictionary JSON file")
    
    # PI task parameters
    parser.add_argument("--n_sessions", type=int, default=5,
                        help="Number of sessions to run")
    
    # New parameters for tracked and untracked keys
    parser.add_argument("--n_tracked_keys", type=int, default=3,
                        help="Number of keys to track and query about")
    parser.add_argument("--n_untracked_keys", type=int, default=0,
                        help="Number of keys to include but not track, if n_total_keys is not 0, this must be set to 0")
    parser.add_argument("--n_total_keys", type=int, default=0,
                        help="Number of total keys to include. If not 0, n_untracked_keys = n_total_keys - n_tracked_keys")  

    parser.add_argument("--n_tracked_updates", type=int, default=3,
                        help="Number of updates for tracked keys")
    parser.add_argument("--n_untracked_updates", type=int, default=3,
                        help="Number of updates for untracked keys")
        
    # Other parameters
    parser.add_argument("--random_update", type=int, default=1,
                        help="Whether to randomly permute the pairs, if 0, first untracked keys are updated, then tracked keys")
    parser.add_argument("--balanced_sample", type=int, default=1,
                        help="Whether use occurrence probability to balance the sampling across different sessions")
    parser.add_argument("--memory_limit", type=int, default=1,
                        help="Clear chat history every N sessions (1=no memory)")
    parser.add_argument("--probe_target", type=str, default="current",
                    help="influence which update to probe and how it is probed")
    parser.add_argument("--prompt_updating", type=str, default="colon",
                    help="systematic formating of basic updating style: colon, hyphen, equal, wordy")
    parser.add_argument("--prompt_forgetting", type=str, default="none",
                    help="Whether to prompt the model to forget at each update, full form is like \
                        pos0 options: none, sayreset, hackreset, fwdattend \
                        pos1 options: none, half, full, ; \
                        pos2 options: float-like, fractions, n or -n, OR absolute location using [n], [-n]; \
                        pos3 options: separate, together \
                        e.g. half_1/3_separate, means half of the tracked keys are forgotten separately right after their own 1/3 updating locations ")
    parser.add_argument("--response_format", type=str, default="verbal_succinct",
                    help="response as a single dictionary of multiple sentences, use dict or verbal")
    parser.add_argument("--sample_replacement", type=int, default=0,
                    help="Whether to replace the item at each update")
    parser.add_argument("--remix_category", type=int, default=0,
                    help="Whether to remix the category evenl; for testing categorial influence")
    
    parser.add_argument("--lengthen_item", type=str, default="1_none",
                    help="Parameters for creating longer items by sampling multiple rounds and concatenating them. \
                        Format: 'rounds_style' where rounds is the number of sampling rounds, \
                        and style is one of: {'none', 'strip', 'cap' ,'cap-strip', 'hyphen-strip', 'underscore-strip', 'hyphen-cap-strip', 'underscore-cap-strip'}. \
                        If only one parameter is given, style defaults to 'none'.")


    ## the argument lengthen_item has been separated into two arguments, if it is set to deprecated
    parser.add_argument("--len_item", type=int, default=1,
                    help="Length of the item to be updated")
    parser.add_argument("--len_item_style", type=str, default="none",
                    help="Style of the item to be updated, none, strip, cap, hyphen-strip, underscore-strip, hyphen-cap-strip, underscore-cap-strip")
    
    ## hack parameters
    args = parser.parse_args()

    if args.model_name.startswith("grok-3"):
        ## cot models, including grok-3-mini-beta and grok-3-beta
        args.max_tokens = 10000

    # fix max token for various models, if model name ends with thinking, it is a CoT model
    if args.model_name.startswith("gemini-2.5-pro") or args.model_name.startswith("gemini-2.5-flash") or ("thinking" in args.model_name):
        args.max_tokens = None

    # directly stop a few CoT models to prevent intermediate output overflow as they just count the updates one by one
    list_early_stop_models_prefix = ["gemini-2.0-flash-thinking", "gemini-2.5-pro", "gemini-2.5-flash", "deepseek-reasoner",]

    if args.n_tracked_keys * args.n_tracked_updates > 4600:
        for m in list_early_stop_models_prefix:
            ## if the model name ends with -thinking, it is a CoT model
            if args.model_name.startswith(m) and args.model_name.endswith("thinking"):
                ## directly stop the test
                raise ValueError(f"STOP before starting to prevent intermediate output overflow: Model {args.model_name}, a dumb CoT that counts the updates one by one")

    # Run the PI test
    run_pi_test(args)

if __name__ == "__main__":
    main() 