import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
import json
import matplotlib.pyplot as plt
import re
import warnings
from scipy.stats import entropy


def compute_optimal_grid(n_panels, max_n_cols):
    """
    Compute the optimal number of rows and columns for subplot layout.
    
    Parameters:
    -----------
    n_panels : int
        Number of panels/subplots needed
    max_n_cols : int
        Maximum number of columns allowed
    
    Returns:
    --------
    tuple
        (n_rows, n_cols)
    
    Notes:
    ------
    Optimization principles:
    1. n_cols must be between 2 and max_n_cols
    2. Minimize number of unused axes
    3. Minimize number of rows when possible
    """
    # Ensure max_n_cols is at least 2
    max_n_cols = max(2, max_n_cols)
    
    # Initialize with maximum waste scenario
    best_n_cols = 2
    best_n_rows = (n_panels + 1) // 2
    min_waste = best_n_rows * 2 - n_panels
    
    # Try all possible column counts from 2 to max_n_cols
    for n_cols in range(2, max_n_cols + 1):
        # Calculate rows needed (ceiling division)
        n_rows = (n_panels + n_cols - 1) // n_cols
        
        # Calculate wasted cells
        waste = n_rows * n_cols - n_panels
        
        # Update best configuration if waste is less
        # or if waste is equal but fewer rows
        if waste < min_waste or (waste == min_waste and n_rows < best_n_rows):
            min_waste = waste
            best_n_cols = n_cols
            best_n_rows = n_rows
    
    return best_n_rows, best_n_cols 


def normalized_entropy(p, base=2):
    """
    Compute the normalized Shannon entropy of a discrete distribution p.
    
    Parameters
    ----------
    p : array-like
        Probability vector (must sum to 1).  Zero-probability entries are ignored.
    base : {2, e, 10}, optional
        Logarithm base for entropy.  If 2, entropy is in bits; if e, in nats, etc.
    
    Returns
    -------
    H_norm : float
        Normalized entropy in [0,1], where 1 means uniform.
    """

    ## if p is a list convert it to an array
    if isinstance(p, list):
        p = np.array(p, dtype=float)

    if not np.isclose(p.sum(), 1):
        raise ValueError("p must sum to 1")
    # filter out zeros to avoid 0⋅log0
    p_nonzero = p[p > 0]

    # raw entropy
    H = entropy(p_nonzero, base=base)

    if H == 0:
        return 0.0

    # maximum entropy for total number of states (including zeros)
    n_total = p.size  # Use total size of input vector, not just non-zero
    H_max = np.log(n_total) / (np.log(base) if base != np.e else 1)
    return H / H_max


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


def find_run_files(P,lst_patterns):
    if not isinstance(P,Path):
        P = Path(P)

    lst_files = []
    for pattern in lst_patterns:
        files = list(P.glob(pattern))
        
        ## check if file has length 1
        if len(files) == 1:
            lst_files.append(files[0])
        else:
            raise ValueError(f"Found {len(files)} files (should be 1) for pattern {pattern} with {P}")
    return lst_files


def quick_accuracy_analysis_one_expm(path_expm_results,mark_params,nstd_range=None,calc_resp_pos=False,calc_resp_entropy=False):
    lst_folder_patterns = ["*correct-answers.json","*responses.json","*meta-info.json","*all-pairs.json"]
    list_result_runs = []
    list_entropy_results = [] ## entropy is calculated using data from all sessions in a folder
    dict_list_all_pos = {}

    for folder in Path(path_expm_results).glob('*'):

        ## exclude the folder named 'log'
        if 'log' in folder.name:
            continue

        if not folder.is_dir():
            continue

        lst_files = find_run_files(folder,lst_folder_patterns)
        if len(lst_files) != len(lst_folder_patterns):
            raise ValueError(f"Found {len(lst_files)} files (should be {len(lst_folder_patterns)}) for {folder.name}")
        
        list_dict_correct_answers = json.load(open(lst_files[0]))
        list_dict_responses = json.load(open(lst_files[1]))
        dict_meta_info = json.load(open(lst_files[2]))

        if (calc_resp_pos) or (calc_resp_entropy):
            nested_list_all_pairs = json.load(open(lst_files[3]))

        mark_params_values = [dict_meta_info[param] for param in mark_params]

        ## calculate accuracy
        for i, (dict_responses, dict_correct_answers) in enumerate(zip(list_dict_responses, list_dict_correct_answers)):
            accuracy, n_missing = compute_accuracy(dict_responses, dict_correct_answers)
            result_one_session = mark_params_values + [i] + [accuracy, n_missing]
            list_result_runs.append(result_one_session)

        ## get all response pos in stream
        if (calc_resp_pos) or (calc_resp_entropy):
            list_all_pos = extract_all_response_pos_in_stream(nested_list_all_pairs,list_dict_responses,flatten=True)
    
            n_updates = dict_meta_info['n_tracked_updates']
            key_now = f"{mark_params[0]}-{mark_params_values[0]}&{mark_params[1]}-{mark_params_values[1]}"
            dict_list_all_pos[key_now] = (list_all_pos,n_updates) ## return a tuple of list_all_pos and n_updates
        else:
            dict_list_all_pos = None

        if calc_resp_entropy:
            list_all_pos_deambiguous,frac_replaced = random_sample_ambiguous_responses(list_all_pos,n_updates)
            density, bin_edges = np.histogram(list_all_pos_deambiguous, bins=np.arange(-0.5, n_updates+.5), density=True)
            normal_entropy = normalized_entropy(density,base=2)
            list_entropy_results.append(mark_params_values + [normal_entropy,frac_replaced])


    df_result = pd.DataFrame(list_result_runs,columns= mark_params + ["session"] + ["accuracy","n_missing"])
    df_result.sort_values(by=mark_params + ["session"],inplace=True)

    if calc_resp_entropy:
        df_result_entropy = pd.DataFrame(list_entropy_results,columns=mark_params + ["normal_entropy","frac_resampled_values"])
        return df_result,dict_list_all_pos,df_result_entropy
    else:
        return df_result,dict_list_all_pos




def random_sample_ambiguous_responses(list_all_pos,n_updates):

    ## -1: correct key, but out of updates answer, resample for this
    ## -2: missing key, do not resample for this
    ## return the number of replaced values (values that are smaller than 0)
    ## frac_amb is the fraction of -2 in the list_all_pos

    frac_missing_keys = sum(1 for pos in list_all_pos if pos == -2)/len(list_all_pos)

    ## remove all -2 in the list_all_pos
    list_all_pos_replaced = [pos for pos in list_all_pos if pos != -2]

    ## resample the -1 with a random number between 0 and n_updates-1
    list_all_pos_replaced = [np.random.randint(0, n_updates) if pos == -1 else pos for pos in list_all_pos_replaced]

    return list_all_pos_replaced,frac_missing_keys


def isolate_ambiguous_responses(list_all_pos):
    """
    Isolate the ambiguous responses from the list of response positions.
    """
    list_unambiguous_pos = []
    amb_pos_count = 0
    for pos in list_all_pos:
        if pos < 0:
            amb_pos_count += 1
        else:
            list_unambiguous_pos.append(pos)

    frac_amb = amb_pos_count / len(list_all_pos)
    return list_unambiguous_pos,frac_amb


def filter_out_outliers(df_result,mark_params,n_std_range):
    list_df_result = []
    list_report = []
    for ids,dfw in df_result.groupby(mark_params):
        ## filter out outliers according to the accuracy column
        mean_accuracy = np.mean(dfw["accuracy"])
        std_accuracy = np.std(dfw["accuracy"])
        n_sessions_before = len(dfw)
        dfw = dfw[dfw["accuracy"] > mean_accuracy - n_std_range * std_accuracy]
        dfw = dfw[dfw["accuracy"] < mean_accuracy + n_std_range * std_accuracy]
        n_sessions_after = len(dfw)
        list_df_result.append(dfw)
        list_report.append(list(ids) + [n_sessions_before,n_sessions_after,n_sessions_after-n_sessions_before])
    df_result = pd.concat(list_df_result)
    df_report = pd.DataFrame(list_report,columns=mark_params + ["n_sessions_before","n_sessions_after","n_sessions_removed"])
    return df_result,df_report


def extract_update_sequence_by_key(list_all_pairs,key):
    dict_stream_updates = {}
    for pair in list_all_pairs:
        key,value = pair
        if key not in dict_stream_updates:
            dict_stream_updates[key] = []
        dict_stream_updates[key].append(value)
    return dict_stream_updates


def get_response_pos_in_stream(dict_stream_updates,dict_response):
    """
    Get the response positions in the stream.
    The order of the keys are determined by the order of the keys in dict_stream_updates.
    """
    if dict_response is None:
        dict_response = {}
    list_pos = []
    list_keys = []
    for key, value in dict_stream_updates.items():
        if key in dict_response:
            try:
                list_pos.append(value.index(dict_response[key]))
            except ValueError:
                ## The answer is not in the stream (out of set answer)
                list_pos.append(-1)
        else:
            ## not answered key
            list_pos.append(-2)
        list_keys.append(key)
    return list_pos,list_keys


def mark_keys_by_occurence_order(lst_pairs,target_keys=[]):
    """
    Mark the keys in the order of occurrence in the stream.
    If target_keys is provided, only mark the keys in target_keys.
    """
    lst_key_order = []
    if len(target_keys) > 0:
        for pair in lst_pairs:
            key,value = pair
            if (key not in lst_key_order) and (key in target_keys):
                lst_key_order.append(key)
            ## break if all the target keys are marked
            if len(lst_key_order) == len(target_keys):
                break
    else:
        for pair in lst_pairs:
            key,value = pair
            if key not in lst_key_order:
                lst_key_order.append(key)
    return lst_key_order


def extract_one_session_response_pos_in_stream(dict_stream_updates,dict_response,lst_keys_tomark):

    if dict_response is None:
        ## no response, return empty lists
        return [],[]
    
    lst_pos,lst_keys = get_response_pos_in_stream(dict_stream_updates,dict_response)
    
    if lst_keys_tomark:
        lst_pos_marked = []
        lst_pos_notmarked = []

        ## separate the lst_pos into list_all_pos_marked and list_all_pos_notmarked
        ## by whether the key is in list_dict_keys_tomark
        for key,pos in zip(lst_keys,lst_pos):
            if key in lst_keys_tomark:
                lst_pos_marked.append(pos)
            else:
                lst_pos_notmarked.append(pos)
    else:
        ## simplify the computation
        lst_pos_marked = lst_pos
        lst_pos_notmarked = []
    
    return lst_pos_marked,lst_pos_notmarked


def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def extract_all_response_pos_in_stream(nested_list_all_pairs,list_dict_responses,list_dict_correct_answers,nested_list_keys_tomark=[],flatten=False,resp_order="none"):
    """
    Extract the response positions in the stream for all pairs.
    resp_order: "query" or "update" or "both"
    """
    ## check if resp_order is valid
    if resp_order not in ["none","query","update","both"]:
        raise ValueError(f"Invalid resp_order: {resp_order}, please use {resp_order}")

    list_all_pos_marked_query_order = []
    list_all_pos_notmarked_query_order = []

    if resp_order in ["update","both"]:
        list_all_pos_marked_update_order = []
        list_all_pos_notmarked_update_order = []
        

    return_package = []
    for lst_pairs,dict_response,dict_correct_answers,lst_keys_tomark in zip(nested_list_all_pairs,list_dict_responses,list_dict_correct_answers,nested_list_keys_tomark):

        ## skip if no response
        if dict_response is None:
            continue

        dict_stream_updates = extract_update_sequence_by_key(lst_pairs,"tracked_key")

        ## reorder the dict_stream_updates according to the lst_keys_update_order
        if resp_order in ["query","both"]:
            lst_keys_query_order = list(dict_correct_answers.keys())
            dict_stream_updates_query_order = {key: dict_stream_updates[key] for key in lst_keys_query_order}
            lst_pos_marked_query_order,lst_pos_notmarked_query_order = extract_one_session_response_pos_in_stream(dict_stream_updates_query_order,dict_response,lst_keys_tomark)
            list_all_pos_marked_query_order.append(lst_pos_marked_query_order)
            list_all_pos_notmarked_query_order.append(lst_pos_notmarked_query_order)

        if resp_order in ["update","both"]:
            lst_keys_query_order = list(dict_correct_answers.keys())
            lst_keys_update_order = mark_keys_by_occurence_order(lst_pairs,lst_keys_query_order)
            dict_stream_updates_update_order = {key: dict_stream_updates[key] for key in lst_keys_update_order}
            lst_pos_marked_update_order,lst_pos_notmarked_update_order = extract_one_session_response_pos_in_stream(dict_stream_updates_update_order,dict_response,lst_keys_tomark)
            list_all_pos_marked_update_order.append(lst_pos_marked_update_order)
            list_all_pos_notmarked_update_order.append(lst_pos_notmarked_update_order)

        if resp_order == "none":
            ## no reordering
            lst_pos_marked_none_order,lst_pos_notmarked_none_order = extract_one_session_response_pos_in_stream(dict_stream_updates,dict_response,lst_keys_tomark)
            list_all_pos_marked_query_order.append(lst_pos_marked_none_order)
            list_all_pos_notmarked_query_order.append(lst_pos_notmarked_none_order)

    if resp_order in ["none","query","both"]:
        return_package.append(list_all_pos_marked_query_order)
        return_package.append(list_all_pos_notmarked_query_order)
    if resp_order in ["update","both"]:
        return_package.append(list_all_pos_marked_update_order)
        return_package.append(list_all_pos_notmarked_update_order)

    if flatten:
        return_package = list(map(flatten_list_of_lists,return_package))
    
    return return_package


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


def extract_after_first_colon(json_string):
    """
    Extract the content after the first colon in a JSON string.
    
    Args:
        json_string: A string containing JSON with at least one colon
        
    Returns:
        The content after the first colon (including all subsequent text)
    """
    # Find the position of the first colon
    colon_index = json_string.find(':')
    
    if colon_index != -1:
        # Return everything after the first colon (skipping the colon itself)
        return json_string[colon_index + 1:].strip()
    else:
        return None


def add_a_pos_distribution_plot(list_resp_pos, n_updates, ax, 
                normalization=None, n_err_bins=0, inverse_pos=False, 
                to_frac=False, fs=8, vertical=False):
    """
    Plot a histogram of response positions with customizable binning.
    
    Parameters:
    -----------
    list_resp_pos : array-like
        List of response positions to plot
    n_updates : int
        Number of updates in the sequence
    ax : matplotlib.axes.Axes
        Axes to plot on
    normalization : str, default=None
        if none, return count
        if "density", return density (uneven bins are normalized differently)
        if "binwise", all bins sum up to one
    n_err_bins : int, default=0
        If > 0, positions 0 to n_updates-2 will be grouped into n_err_bins bins,
        while positions -2, -1, and n_updates-1 remain as separate bins.
        If 0, every position is treated as a separate bin (original behavior).
    vertical : bool, default=False
        If True, displays the bars vertically (horizontal bars). If False, displays horizontal bars.
    
    Returns:
    --------
    counts : array
        The counts or density in each bin
    bin_centers : array
        The centers of each bin
    """

    if normalization is None:
        to_density = False
    elif normalization == "density":
        to_density = True
    elif normalization == "binwise":
        to_density = False

    ## remove nan values from list_resp_pos
    list_resp_pos = [v for v in list_resp_pos if not np.isnan(v)]

    bar_width = 0.9
    if n_err_bins <= 0:
        # Original behavior: treat every position as a separate bin

        list_resp_pos_copy = []
        for pos in list_resp_pos:
            if pos >= 0:
                list_resp_pos_copy.append(pos+1)
            else:
                list_resp_pos_copy.append(pos)
        list_resp_pos = list_resp_pos_copy

        counts, bin_edges = np.histogram(list_resp_pos, bins=np.arange(-2.5, n_updates+1.5), density=to_density)

        if normalization == "binwise":
            counts = counts / np.sum(counts)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_indices = bin_centers

        if to_frac:
            bin_indices = bin_indices / (n_updates-1)
            bar_width = 0.9 / (n_updates-1)
    else:
        # Custom binning: separate bins for positions -2, -1, and n_updates-1, 
        # while grouping positions 0 to n_updates-2 into n_err_bins bins
        
        # Create special bins for positions -2, -1, and the last position
        special_bins = [-2.5, -1.5]
        
        # Create evenly spaced bins for the positions 0 to n_updates-2
        regular_range = np.linspace(-0.5, n_updates-1.5, n_err_bins+1)
        
        # Add the last position bin
        all_bins = np.concatenate([special_bins, regular_range, [n_updates-0.5]])
        
        counts, bin_edges = np.histogram(list_resp_pos, bins=all_bins, density=to_density)

        if normalization == "binwise":
            counts = counts / np.sum(counts)
        
        # Calculate bin centers, for xtick labels only
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        ## use the bin indices as xticks
        bin_indices = np.arange(len(bin_centers))

        ## bin_names as f"{start}-{end}"; fix python starting from 0; here we start from 1
        if inverse_pos:
            regular_range = regular_range[::-1]
            regular_bin_names = [f"{int(np.ceil(j+1))}-{int(np.ceil(i+1))}" for i,j in zip(regular_range[:-1],regular_range[1:])]
            bin_names = ["off keys","off values"] + regular_bin_names + ["final"]
        else:
            regular_bin_names = [f"{int(np.ceil(i+1))}-{int(np.ceil(j+1))}" for i,j in zip(regular_range[:-1],regular_range[1:])]
            bin_names = ["off keys","off values"] + regular_bin_names + [f"{n_updates}"]
        
    ## set colors
    colors = ["dimgray","darkgray"] + ["teal"] * (len(bin_indices)-3) + ["tan"]
    
    # Plot the histogram
    if vertical:
        ax.barh(bin_indices, counts, height=bar_width, color=colors, align='center')
    else:
        ax.bar(bin_indices, counts, width=bar_width, color=colors, align='center')

    if n_err_bins > 0:
        if vertical:
            ax.set_yticks(bin_indices)
            ax.set_yticklabels(bin_names, fontsize=fs)
        else:
            ax.set_xticks(bin_indices)
            ax.set_xticklabels(bin_names, fontsize=fs, rotation=45, ha='right', va='top')  # align to the top right

    return counts, bin_centers 


def quick_accuracy_analysis_one_expm_multiruns(path_expm_results, mark_params, nstd_range=None, calc_resp_pos=False, resp_order="query",calc_resp_entropy=False, mark_forget=False):
    """
    Similar to quick_accuracy_analysis_one_expm but groups sibling folders that share parameter values.
    Treats multiple runs with the same parameters as a single merged run.
    
    Args:
        path_expm_results (str): Path to the experiment results.
        mark_params (list): List of parameters to mark in the result dataframe.
        nstd_range (float, optional): Range for filtering outliers. Defaults to None.
        calc_resp_pos (bool, optional): Whether to calculate response positions. Defaults to False.
        calc_resp_entropy (bool, optional): Whether to calculate response entropy. Defaults to False.
        
    Returns:
        tuple: (df_result, dict_list_all_pos, df_result_entropy) similar to quick_accuracy_analysis_one_expm
    """
    lst_folder_patterns = ["*correct-answers.json", "*responses.json","*meta-info.json", "*all-pairs.json", "*forget.json"]
    list_result_runs = []
    list_entropy_results = []  # entropy is calculated using data from all sessions in a group
    dict_list_all_pos_marked = {}
    dict_list_all_pos_notmarked = {}
    
    # Group folders by parameters (ignoring date_time prefix)
    folder_groups = {}
    
    for folder in Path(path_expm_results).glob('*'):
        # Exclude the folder named 'log'
        if 'log' in folder.name:
            continue
            
        if not folder.is_dir():
            continue
            
        # Extract date_time and parameters from folder name
        # Pattern: date_time_param1-value1_param2-value2...
        folder_name = folder.name
        match = re.match(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(.*)', folder_name)
        
        if match:
            date_time = match.group(1)
            params_part = match.group(2)
        else:
            # If no date_time prefix, use the whole name as params_part
            date_time = ""
            params_part = folder_name
            
        # Use params_part as the group key
        if params_part not in folder_groups:
            folder_groups[params_part] = []
            
        folder_groups[params_part].append((date_time, folder))
    
    print("processing ", str(folder))
    
    # Process each group of folders
    for params_part, folder_list in folder_groups.items():
        # Sort folders by date_time
        folder_list.sort(key=lambda x: x[0])
        
        # Initialize merged data structures
        global_list_dict_correct_answers = []
        global_list_dict_responses = []
        global_list_forget = []
        global_nested_list_all_pairs = []
        dict_meta_info_latest = None
        
        # Process each folder in the sorted group
        for _, folder in folder_list:


            ## unmute this to check a specific folder
            # folder_to_check = "2025-05-02_20-52-01_ntrk-46_ntrkupd-281"
            # if folder.name != folder_to_check:
            #     continue

            # Find required files
            try:
                lst_files = find_run_files(folder, lst_folder_patterns)
                if len(lst_files) != len(lst_folder_patterns):
                    warnings.warn(f"Found {len(lst_files)} files (should be {len(lst_folder_patterns)}) for {folder.name}, skipping")
                    continue
                    
                # Load data
                list_dict_correct_answers = json.load(open(lst_files[0]))
                list_dict_responses = json.load(open(lst_files[1]))
                dict_meta_info = json.load(open(lst_files[2]))
                
                # Verify both lists have the same length
                if len(list_dict_correct_answers) != len(list_dict_responses):
                    warnings.warn(f"list_dict_correct_answers and list_dict_responses have different lengths in {folder.name}, skipping")
                    continue
                    
                # Merge data
                global_list_dict_correct_answers.extend(list_dict_correct_answers)
                global_list_dict_responses.extend(list_dict_responses)
                
                # Update meta_info to the latest one
                dict_meta_info_latest = dict_meta_info
                
                # Handle pairs data if needed
                if (calc_resp_pos) or (calc_resp_entropy):
                    nested_list_all_pairs = json.load(open(lst_files[3]))
                    global_nested_list_all_pairs.extend(nested_list_all_pairs)

                    if mark_forget:
                        ## also load the forget data
                        lst_forget = json.load(open(lst_files[4]))
                        global_list_forget.extend(lst_forget)
                    else:
                        global_list_forget.extend([[]]*len(nested_list_all_pairs))
                    
            except Exception as e:
                warnings.warn(f"Error processing folder {folder.name}: {str(e)}")
                continue
        
        # Skip if no valid data was found
        if not global_list_dict_correct_answers or dict_meta_info_latest is None:
            continue
            
        # Extract parameter values from meta_info
        mark_params_values = [dict_meta_info_latest[param] for param in mark_params]
        
        # Calculate accuracy for each session
        for i, (dict_responses, dict_correct_answers) in enumerate(zip(global_list_dict_responses, global_list_dict_correct_answers)):
            accuracy, n_missing = compute_accuracy(dict_responses, dict_correct_answers)
            result_one_session = mark_params_values + [i] + [accuracy, n_missing]
            list_result_runs.append(result_one_session)
        
        # Process response positions if needed
        if (calc_resp_pos) or (calc_resp_entropy):
            list_all_pos_marked,list_all_pos_notmarked = extract_all_response_pos_in_stream(global_nested_list_all_pairs, global_list_dict_responses, global_list_dict_correct_answers, global_list_forget, 
                                                                                            flatten=True, resp_order=resp_order)
            
            n_updates = dict_meta_info_latest['n_tracked_updates']
            key_now = f"{mark_params[0]}-{mark_params_values[0]}&{mark_params[1]}-{mark_params_values[1]}"
            dict_list_all_pos_marked[key_now] = (list_all_pos_marked, n_updates)  # return a tuple of list_all_pos and n_updates
            dict_list_all_pos_notmarked[key_now] = (list_all_pos_notmarked, n_updates)  # return a tuple of list_all_pos and n_updates
        else:
            dict_list_all_pos_marked = None
            dict_list_all_pos_notmarked = None
            
        # Calculate entropy if needed
        if calc_resp_entropy:
            list_all_pos_deambiguous, frac_amb = random_sample_ambiguous_responses(list_all_pos_marked, n_updates)

            ## calculate the normalized entropy of the deambiguous list
            ## at least 20 deambiguous responses are required for this, otherwise return 0
            if len(list_all_pos_deambiguous) < 20:
                normal_entropy = np.nan
            else:
                density, bin_edges = np.histogram(list_all_pos_deambiguous, bins=np.arange(-0.5, n_updates+.5), density=True)
                normal_entropy = normalized_entropy(density, base=2)
            list_entropy_results.append(mark_params_values + [normal_entropy, frac_amb])
    
    # Create result dataframes
    df_result = pd.DataFrame(list_result_runs, columns=mark_params + ["session"] + ["accuracy", "n_missing"])
    df_result.sort_values(by=mark_params + ["session"], inplace=True)
    
    if calc_resp_entropy:
        df_result_entropy = pd.DataFrame(list_entropy_results, columns=mark_params + ["normal_entropy", "frac_ambiguous"])
    else:
        df_result_entropy = pd.DataFrame()

    if mark_forget:
        return df_result, (dict_list_all_pos_marked, dict_list_all_pos_notmarked), df_result_entropy
    else:
        return df_result, dict_list_all_pos_marked, df_result_entropy



if __name__ == "__main__":

    ## load a dataset to test quick_accuracy_analysis_one_expm_multiruns
    # path_expm_results = "/Users/vince/Documents/LLM_AttentionBenchmark/eval_pi/test_updates/deepseek-chat-test"
    # # df_result, (dict_list_all_pos_marked, dict_list_all_pos_notmarked), df_result_entropy = quick_accuracy_analysis_one_expm_multiruns(path_expm_results, ["model_name","prompt_forgetting","n_tracked_updates"], calc_resp_pos=True, mark_forget=False)
    # df_result, dict_list_all_pos, df_result_entropy = quick_accuracy_analysis_one_expm_multiruns(path_expm_results, ["model_name","prompt_forgetting","n_tracked_updates"], calc_resp_pos=True, mark_forget=False, calc_resp_entropy=True)
    # print(df_result)


    # response = "animal: pig; color: red; shape: square; Here are the current values. "
    # print(_extract_colon_matches(response))
    
    # a  = normalized_entropy(np.array([1,0,0]))
    # print(a)

    # b  = normalized_entropy(np.array([0.25,0.25,0.25,0.25]))
    # print(b)


    # # Generate fake data
    # np.random.seed(42)
    # n_updates = 10  # Total number of updates in the sequence

    # # Create response positions with a mix of valid positions and error codes
    # valid_positions = np.random.randint(0, n_updates, size=80)  # Regular positions
    # error_positions_1 = np.full(10, -1)  # Off-items errors (value -1)
    # error_positions_2 = np.full(5, -2)  # Off-keys errors (value -2)
    # list_resp_pos = np.concatenate([valid_positions, error_positions_1, error_positions_2])

    # # Create a figure with several subplots to show different ways to use the function
    # fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # # Plot 1: Basic histogram with default settings
    # add_a_pos_distribution_plot(
    #     list_resp_pos=list_resp_pos,
    #     n_updates=n_updates,
    #     ax=axes[0, 0],
    #     normalization=None  # Show raw counts
    # )
    # axes[0, 0].set_title("Raw counts, individual bins")
    # axes[0, 0].set_xlabel("Position")
    # axes[0, 0].set_ylabel("Count")

    # # Plot 2: 
    # add_a_pos_distribution_plot(
    #     list_resp_pos=list_resp_pos,
    #     n_updates=n_updates,
    #     ax=axes[0, 1],
    #     normalization=None,  
    #     n_err_bins=3
    # )
    # axes[0, 1].set_title(" x ")
    # axes[0, 1].set_xlabel("Position")
    # axes[0, 1].set_ylabel("Count")

    # # Plot 3: Using grouped bins with n_err_bins
    # add_a_pos_distribution_plot(
    #     list_resp_pos=list_resp_pos,
    #     n_updates=n_updates,
    #     ax=axes[1, 0],
    #     normalization="binwise",  # All bins sum to 1
    #     n_err_bins=3  # Group positions into 3 bins
    # )
    # axes[1, 0].set_title("Binwise normalization with 3 error bins")
    # axes[1, 0].set_xlabel("Position bins")
    # axes[1, 0].set_ylabel("Fraction")


    # # Plot 4: Using grouped bins with n_err_bins
    # add_a_pos_distribution_plot(
    #     list_resp_pos=list_resp_pos,
    #     n_updates=n_updates,
    #     ax=axes[1, 1],
    #     normalization="density",  # All bins sum to 1
    #     n_err_bins=3  # Group positions into 3 bins
    # )
    # axes[1, 1].set_title("Binwise normalization with 3 error bins")
    # axes[1, 1].set_xlabel("Position bins")
    # axes[1, 1].set_ylabel("Fraction")


    # plt.tight_layout()
    # # plt.savefig("position_distribution_plots.png")
    # plt.show()
    root_path = ""
    eval_root = f"{root_path}eval_pi/"
    test_name = "test_updates"
    mark_params = ["n_tracked_keys","n_tracked_updates","finished_sessions"]
    model_name = "claude-3-5-sonnet"
    path_test_results = f"{eval_root}{test_name}/"
    path_expm_results = f"{path_test_results}{model_name}/"
    df_result, dict_list_all_pos, df_result_entropy = quick_accuracy_analysis_one_expm_multiruns(path_expm_results, ["model_name","prompt_forgetting","n_tracked_updates"], calc_resp_pos=True, resp_order="update",
                                                                                                 mark_forget=False, calc_resp_entropy=False)
    print(df_result)
    print(dict_list_all_pos)
    print(df_result_entropy)