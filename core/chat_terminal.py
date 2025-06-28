import os
import json
import time
import pickle
import argparse
import warnings
import tkinter as tk
from tkinter import scrolledtext, filedialog, ttk
from datetime import datetime

# Import necessary libraries for different models
import openai
from openai import OpenAI
import anthropic
import google.generativeai as genai
import copy
import subprocess
import requests  # For direct API calls to Vertex AI
import tiktoken

# Import necessary libraries for Google auth
from google.auth import default, transport
from google.oauth2 import service_account

class TokenCounter:
    """Count tokens for different model families"""
    def __init__(self, model_name, client=None):
        self.model_name = model_name
        self.model_family = judge_model_family(model_name)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.enc = tiktoken.encoding_for_model("gpt-4o")
            
        if self.model_family == "gemini":
            if not isinstance(client, genai.GenerativeModel):
                raise ValueError("Client must be of type genai.GenerativeModel")
            self.client = client
    
    def count_text(self, text):
        """Count tokens in text based on model family"""
        # print("--------------------------------")
        # print("type of text: ", type(text))
        # print("text: ", text)
        # print("--------------------------------")
        if text is None:
            return 0
        elif isinstance(text, str):
            if len(text) == 0:
                return 0
        else:
            raise ValueError(f"Unexpected text type: {type(text)}")
        
        if self.model_family == "gemini":
            return self.client.count_tokens(text).total_tokens
    
        else:
            ### approximation for all other models using tiktoken from openai
            return len(self.enc.encode(text))
        

    def add_input_tokens(self, text):
        """Count and add input tokens"""
        count = self.count_text(text)
        self.total_input_tokens += count
        return count
        
    def add_output_tokens(self, text):
        """Count and add output tokens"""
        count = self.count_text(text)
        self.total_output_tokens += count
        return count
    
    def reset(self):
        """Reset token counters"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def subtract_input_tokens(self, text):
        """Subtract input tokens (for goback command)"""
        count = self.count_text(text)
        self.total_input_tokens -= count
        return count
        
    def subtract_output_tokens(self, text):
        """Subtract output tokens (for goback command)"""
        count = self.count_text(text)
        self.total_output_tokens -= count
        return count

class ChatHistoryManager:
    """Manage chat history for different model families"""
    def __init__(self, model_name, output_dir="chat_logs", nick_name="chat"):
        self.model_name = model_name
        self.model_family = judge_model_family(model_name)
        self.history = []
        self.session_count = 0
        self.output_dir = output_dir
        self.base_filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{nick_name}"
        self.snapshot_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Role name mapping for different models
        if self.model_family in ["gpt","llama","mistral","nvidia"]:
            self.role_legal_name_dict = {"user": "user", "model": "assistant"}
        elif self.model_family == "gemini":
            self.role_legal_name_dict = {"user": "user", "model": "model"}
        else:
            self.role_legal_name_dict = {"user": "user", "model": "assistant"}  # undefined model family
            
        # 设置终端以允许系统粘贴
        os.environ.setdefault('TERM', 'xterm-256color')
        
        # 如果使用 curses
        if hasattr(self, 'stdscr'):
            import curses
            curses.meta(1)  # 允许 8 位输入
            self.stdscr.keypad(1)  # 启用特殊键处理
            
    def add_user_message(self, text):
        """Add a user message to history"""

        ## all models uses "user" as the key
        self.history.append({"role": self.role_legal_name_dict["user"], "content": text})
        
        self.session_count += 1
        return True
        
    def add_model_message(self, text):
        """Add a model message to history"""
        ## the role_legal_name_dict will translate 'model' to the model role name according to the model_family
        self.history.append({"role": self.role_legal_name_dict["model"], "content": text})
        return True
        
    def save_history(self, is_snapshot=False, export_format="pickle"):
        """Save chat history to pickle or json file"""
        if is_snapshot:
            base_filename = os.path.join(self.output_dir, f"{self.base_filename}_snapshot-{self.snapshot_count}")
            self.snapshot_count += 1
        else:
            base_filename = os.path.join(self.output_dir, f"{self.base_filename}_full")
        
        # Save in requested format
        if export_format.lower() == "json":
            filename = f"{base_filename}_chat.json"
            with open(filename, "w", encoding="utf-8") as f:
                # Format JSON with indentation for readability
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        else:  # Default to pickle
            filename = f"{base_filename}_chat.pkl"
            with open(filename, "wb") as f:
                pickle.dump(self.history, f)
                
        return filename
        
    def load_history(self, filepath):
        """Load chat history from pickle or json file"""
        if not os.path.exists(filepath):
            return False
        
        # Determine file format based on extension
        file_format = os.path.splitext(filepath)[1].lower()
        
        try:
            if file_format == ".json":
                with open(filepath, "r", encoding="utf-8") as f:
                    loaded_history = json.load(f)
            else:  # Default to pickle
                with open(filepath, "rb") as f:
                    loaded_history = pickle.load(f)
            
            # Convert role names if needed
            converted_history = []
            for msg in loaded_history:
                if isinstance(msg, dict) and "role" in msg:
                    new_msg = msg.copy()
                    
                    # Convert from GPT to Gemini
                    if self.model_family == "gemini" and msg["role"] == "assistant":
                        new_msg["role"] = "model"
                    
                    # Convert from Gemini to GPT
                    elif self.model_family in ["gpt","llama", "mistral", "nvidia"] and msg["role"] == "model":
                        new_msg["role"] = "assistant"
                    
                    converted_history.append(new_msg)
                else:
                    converted_history.append(msg)
            
            # Store the converted history
            self.history = converted_history
            
            # Count user messages to determine session count
            user_role = self.role_legal_name_dict["user"]
            self.session_count = sum(1 for msg in converted_history if msg.get("role") == user_role)
            
            return True
            
        except Exception as e:
            print(f"Error loading history from {filepath}: {e}")
            return False
        
    def remove_last_exchange(self):
        """Remove the last exchange (user + model) from history,
        update the total input and output tokens and session count"""
        if len(self.history) >= 2:
            # Remove model and user messages
            self.history = self.history[:-2]
            self.session_count -= 1
            return True
        elif len(self.history) == 1:
            # Only user message exists
            self.history = []
            self.session_count = 0
            return True
        return False
        
    def reset(self):
        """Clear history and reset session count"""
        self.history = []
        self.session_count = 0
        # Generate new base filename with current timestamp
        self.base_filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.base_filename.split('_', 3)[-1]}"
        self.snapshot_count = 0
        return True

def judge_model_family(model_name):
    """Determine model family based on model name"""
    ## check if model_name contain certain string
    if model_name.startswith(("gpt", "deepseek", "o3", "o4", "grok", "claude","qwen")):
        return "gpt"
    elif model_name.startswith("gemini"):
        return "gemini"
    elif model_name.startswith("nvidia"):
        return "nvidia"
    elif model_name.startswith("llama"):
        return "llama"
    elif model_name.startswith("mistral"):
        return "mistral"
    else:
        warnings.warn(f"Model family not explicitly supported: {model_name}")
        return "unknown"


def load_api_keys(path=""):
    """Load API keys from a JSON file"""
    try:
        with open(f"{path}API.json", "r") as f:
            api_keys = json.load(f)
        print(f"API keys loaded successfully from {path}API.json")
        return api_keys
        
    except FileNotFoundError:
        print("Error: API.json file not found. Please create it with your API keys.")
        exit(1)
    except json.JSONDecodeError:
        print("Error: API.json is not valid JSON. Please check its contents.")
        exit(1)
        

def get_client(model_name, api_keys, temperature, max_tokens):
    """Initialize client based on model name"""
    if model_name.startswith(("gpt","o3","o4")):
        openai.api_key = api_keys["OPENAI_API_KEY"]
        client = OpenAI(api_key=api_keys["OPENAI_API_KEY"])
        return client

    elif model_name.startswith("claude"):
        # Use OpenAI client with Claude API endpoint for Claude models
        client = OpenAI(
            api_key=api_keys["Claude_API_KEY"],
            base_url=api_keys["Claude_ENDPOINT"]
        )
        return client

    elif model_name.startswith("deepseek"):
        client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=api_keys["DEEPSEEK_API_KEY"]
        )
        return client

    elif model_name.startswith("grok"):
        client = OpenAI(
            api_key=api_keys["GROK_API_KEY_2"],
            base_url=api_keys["GROK_ENDPOINT"]
        )
        return client

    elif model_name.startswith("qwen"):
        client = OpenAI(
            api_key=api_keys["QWEN"],
            base_url=api_keys["QWEN_ENDPOINT"]
        )
        return client

    elif model_name.startswith("gemini"):
        genai.configure(api_key=api_keys["GEMINI_API_KEY"])
        generation_config = {
            "temperature": temperature,
            "top_p": 1,
            # "top_k": 1, ## removed: COT model does not support and redudant as temp=0
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
        }

        # Configure safety settings for Gemini
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        client = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
        return client
        
    elif model_name.startswith("llama"):
        # Try to use the credential file specified in API.json
        try:
            # Import necessary libraries for Google auth
            import google.auth.transport.requests
            from google.oauth2 import service_account
            
            # Path to credentials file from API.json
            credentials_path = api_keys.get("GOOGLE_APPLICATION_CREDENTIALS")
            
            # Create credentials from service account file
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            
            # Refresh token to ensure it's valid
            auth_req = google.auth.transport.requests.Request()
            credentials.refresh(auth_req)
            
            # Get the token
            token = credentials.token
            
        except Exception as e:
            print(f"Could not use service account, falling back to gcloud: {e}")
            # Fallback to using gcloud command
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                stdout=subprocess.PIPE,
                text=True
            )
            token = result.stdout.strip()
        
        # Check if this is a Llama 4 model and use the specific location
        if "llama-4" in model_name:
            # Use specific Llama 4 settings
            project_id = api_keys.get("GOOGLE_CLOUD_PROJECT_LLAMA4", api_keys.get("GOOGLE_CLOUD_PROJECT"))
            location = api_keys.get("GOOGLE_CLOUD_LOCATION_LLAMA4", "us-east5")
            print(f"Using Llama 4 specific location: {location} for project: {project_id}")
        else:
            # Use default settings for other Llama models
            project_id = api_keys.get("GOOGLE_CLOUD_PROJECT")
            location = api_keys.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        # Configure OpenAI client with Vertex AI endpoint
        client = OpenAI(
            base_url=f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi/chat/completions?",
            api_key=token
        )
        return client
    
    elif model_name.startswith("mistral"):
        # Use gcloud authentication to get access token
        try:
            # Import necessary libraries for Google auth
            import google.auth.transport.requests
            from google.oauth2 import service_account
            
            # Path to credentials file from API.json
            credentials_path = api_keys.get("GOOGLE_APPLICATION_CREDENTIALS")
            
            # Create credentials from service account file
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            
            # Refresh token to ensure it's valid
            auth_req = google.auth.transport.requests.Request()
            credentials.refresh(auth_req)
            
            # Get the token
            token = credentials.token
            
        except Exception as e:
            print(f"Could not use service account, falling back to gcloud: {e}")
            # Fallback to using gcloud command
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                stdout=subprocess.PIPE,
                text=True
            )
            token = result.stdout.strip()
        
        # Use Google Cloud Project ID and location from API.json
        project_id = api_keys.get("GOOGLE_CLOUD_PROJECT")
        location = api_keys.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        # For Mistral models, we'll use the requests library directly
        # Store the token and project details in a client object that we can use later
        client = {
            "token": token,
            "project_id": project_id,
            "location": location,
            "model_id": model_name
        }
        return client

    elif model_name.startswith("nvidia"):
        client = OpenAI(
            base_url=api_keys.get("NVIDIA_API_ENDPOINT"),
            api_key=api_keys.get("NVIDIA_API_KEY")
        )
        return client

    else:
        print(f"Model {model_name} not supported.")
        exit(1)

def call_api(client, model_name, model_family, messages, max_tokens, temperature):
    """Call API based on model family"""
    # Debug output to help diagnose model family issues
    print(f"DEBUG: Processing model: {model_name}, detected family: {model_family}")
    
    if model_name.startswith("o3") or model_name.startswith("o4"):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_completion_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                reasoning_effort="medium",
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling {model_name} API: {e}")
            return f"Error: {e}"
    
    elif model_family == "gpt":
        try:
            # For grok-3-mini-beta, we need to remove presence_penalty and frequency_penalty parameters
            if model_name.startswith("grok-3-mini-beta"):
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens, ## 1000 will not work
                    top_p=1,
                )
            # For Qwen3 models, we need to use streaming mode which is required by the API
            elif "qwen3" in model_name.lower():
                # Check if thinking mode should be enabled (by "-thinking" suffix in model name)
                # Debug output to diagnose suffix detection
                print(f"Debug - Model name: '{model_name}', length: {len(model_name)}")
                print(f"Debug - Last 9 chars: '{model_name[-9:]}', checking against '-thinking'")
                
                enable_thinking = model_name.lower().endswith("-thinking")
                
                # If model name ends with "-thinking", remove it for the actual API call
                api_model_name = model_name
                if enable_thinking:
                    # Remove "-thinking" suffix for the API call
                    api_model_name = model_name[:-9]  # length of "-thinking" is 9
                
                print(f"Using Qwen3 with streaming mode, thinking mode: {enable_thinking}, API model: {api_model_name}")
                
                stream_response = client.chat.completions.create(
                    model=api_model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    stream=True,  # Streaming is required for Qwen3 models
                    extra_body={"enable_thinking": enable_thinking},
                )
                
                # Collect all chunks into a single response
                full_response = ""
                for chunk in stream_response:
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                
                return full_response
            else:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0, ## removed as some models do not support it, default is 0
                    presence_penalty=0, ## removed as some models do not support it, default is 0
                )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling {model_name} API: {e}")
            return f"Error: {e}"
    

    elif model_family == "llama":
        try:
            # Use OpenAI client for Llama models
            completion = client.chat.completions.create(
                model=f"meta/{model_name}",  # Add "meta/" prefix for Vertex AI
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                # Optional Llama Guard safety settings
                extra_body={
                    "extra_body": {
                        "google": {
                            "model_safety_settings": {
                                "enabled": True,
                                "llama_guard_settings": {},
                            }
                        }
                    }
                }
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling {model_name} API: {e}")
            return f"Error: {e}"
    
    elif model_family == "mistral":
        try:            
            # Extract token and project details from client object
            token = client["token"]
            project_id = client["project_id"]
            location = client["location"]
            model_id = client["model_id"]
            
            # Format the URL for Mistral API
            url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/mistralai/models/{model_id}:rawPredict"
                        
            # Prepare the payload
            payload = {
                "model": model_id,
                "messages": messages
            }
            
            # Add temperature and max tokens if provided
            if temperature is not None:
                payload["temperature"] = temperature
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # Make the API call
            print(f"Calling Mistral API: {model_id}")
            response = requests.post(
                url=url,
                headers=headers,
                json=payload
            )
            
            # Process the response
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    return response_data["choices"][0]["message"]["content"]
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error processing Mistral API response: {e}")
                    print(f"Raw response: {response.text}")
                    return f"Error processing response: {e}"
            else:
                print(f"Mistral API request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                return f"API request failed with status code: {response.status_code}"
                
        except Exception as e:
            print(f"Error calling {model_name} API: {e}")
            return f"Error: {e}"
        
    elif model_family == "nvidia":
        try:
            # Debug output for NVIDIA API call
            print(f"DEBUG NVIDIA: Using base URL: {client.base_url}")
            print(f"DEBUG NVIDIA: Full model name being sent: {model_name}")
            
            # Check if thinking mode should be enabled (by "-thinking" suffix in model name)
            enable_thinking = model_name.lower().endswith("-thinking")
            
            # If model name ends with "-thinking", remove it for the actual API call
            api_model_name = model_name
            if enable_thinking:
                # Remove "-thinking" suffix for the actual API call
                api_model_name = model_name[:-9]  # length of "-thinking" is 9
                print(f"NVIDIA thinking mode enabled. Using model: {api_model_name}")
                thinking_mode = "on"
            else:
                thinking_mode = "off"

            ## replace the _ after nvidia in the model name with / to meet nvidia api requirements
            api_model_name = api_model_name.replace("nvidia_", "nvidia/")
            
            # Create a new messages array for NVIDIA API
            nvidia_messages = []
            
            # Add system message at the beginning
            nvidia_messages.append({"role": "system", "content": f"detailed thinking {thinking_mode}"})
            
            # Copy all non-system messages from the original messages
            for msg in messages:
                if msg.get("role") != "system":
                    nvidia_messages.append(msg)
            
            # Use OpenAI client for NVIDIA models
            completion = client.chat.completions.create(
                model=api_model_name,
                messages=nvidia_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return completion.choices[0].message.content
        except Exception as e:
            error_msg = f"Error calling {model_name} API: {e}"
            print(error_msg)
            # Return a more user-friendly error message
            return f"Error: {e}\n\nMake sure the model name format is correct: {model_name}"
        
    elif model_family == "gemini":
        try:
            # For first-time use where there's no chat session yet, create one with "hello"
            if not hasattr(client, '_first_message_template'):
                # Create a temporary chat session to get message structure template
                temp_chat = client.start_chat()
                response = temp_chat.send_message("hello")
                
                # Save the message template
                if hasattr(temp_chat, 'history') and len(temp_chat.history) > 0:
                    client._first_message_template = temp_chat.history[0]
                else:
                    raise Exception("Could not get message template from Gemini API")
            
            # Get the message template
            message_template = client._first_message_template
            
            # Always create a new gemini_history from messages
            gemini_history = []
            
            # Process all messages except the last one (which is the user's query)
            # Assuming messages array has odd length with last entry being user message
            for i in range(len(messages) - 1):
                msg = messages[i]
                if isinstance(msg, dict):
                    # Create a new message entry using the template
                    new_entry = copy.deepcopy(message_template)
                    
                    # Set role (user or model)
                    role = msg.get("role")
                    if role == "assistant":
                        role = "model"
                    new_entry.role = role
                    
                    # Set content
                    content = msg.get("content", "")
                    new_entry.parts[0].text = content
                    
                    # Add to history
                    gemini_history.append(new_entry)
            
            # Create a new chat session with the constructed history
            chat = client.start_chat(history=gemini_history)
            
            # Get the last user message
            last_message = ""
            if messages and messages[-1].get("role") == "user":
                last_message = messages[-1].get("content", "")
            
            # Generate response
            response = chat.send_message(last_message)
            
            # Save the current chat session reference (to keep the message template accessible)
            client._chat_session = chat
            
            return response.text
        except Exception as e:
            print(f"Error calling {model_name} API: {e}")
            return f"Error: {e}"
        
    return "Error: Model not supported for API calls. check call_api function"


def save_meta_info(chatman, token_counter, args, is_snapshot=False):
    """Save metadata to a JSON file with each parameter on a new line"""
    if is_snapshot:
        meta_filename = os.path.join(chatman.output_dir, 
                                    f"{chatman.base_filename}_snapshot-{chatman.snapshot_count-1}_meta.json")
    else:
        meta_filename = os.path.join(chatman.output_dir, f"{chatman.base_filename}_full_meta.json")
    
    meta_info = {
        "model_name": args.model_name,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "session_count": chatman.session_count,
        "total_input_tokens": token_counter.total_input_tokens,
        "total_output_tokens": token_counter.total_output_tokens,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nick_name": args.nick_name
    }
    
    # Write JSON with each parameter on a new line for readability
    with open(meta_filename, "w") as f:
        f.write("{\n")
        # Write all items except the last with a comma
        items = list(meta_info.items())
        for i, (key, value) in enumerate(items):
            if i < len(items) - 1:
                f.write(f'  "{key}": {json.dumps(value)},\n')
            else:
                f.write(f'  "{key}": {json.dumps(value)}\n')
        f.write("}\n")
    
    return meta_filename

def print_colored(text, color_code):
    """Print colored text"""
    print(f"\033[{color_code}m{text}\033[0m")


def try_use_snapshot_model(model_general_name,path_snap_shot_json=None):
    """
    Try to use the snapshot model of the general model name
    The snapshot info is setup in a json file
    """
    if path_snap_shot_json is None:
        print("No snapshot info file provided, use the general model name as the snapshot name")
        return model_general_name

    try:
        with open(path_snap_shot_json, 'r') as f:
            snap_shot_info = json.load(f)
    except FileNotFoundError:
        ## use the general model name as the snapshot name
        print(f"The snapshot info file {path_snap_shot_json} is not found, use the general model name as the snapshot name")
        return model_general_name

    if model_general_name not in snap_shot_info:
        ## use the general model name as the snapshot name
        print(f"The model {model_general_name} is not found in the snapshot info file {path_snap_shot_json}, use the general model name as the snapshot name")
        return model_general_name

    print(f"The model {model_general_name} is found in the snapshot info file {path_snap_shot_json}, use the snapshot model {snap_shot_info[model_general_name]}")
    return snap_shot_info[model_general_name]


class ChatGUI:
    """GUI for chat terminal"""
    def __init__(self, args):
        self.args = args
        self.root = tk.Tk()
        self.root.title(f"Chat with {args.model_name}")
        
        # Set window size and make it resizable
        self.root.geometry("900x700")
        self.root.minsize(600, 400)
        
        # Configure the grid layout
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=0)
        
        # Create conversation display area
        self.conversation_display = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, 
                                                            width=80, height=30)
        self.conversation_display.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self.conversation_display.config(state=tk.DISABLED)
        
        # Add tags for colored text
        self.conversation_display.tag_configure("user", foreground="#008000")  # Green for user
        self.conversation_display.tag_configure("model", foreground="#0000FF")  # Blue for model
        self.conversation_display.tag_configure("info", foreground="#888888")  # Gray for info messages
        self.conversation_display.tag_configure("error", foreground="#FF0000")  # Red for errors
        
        # Create input area frame
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        input_frame.grid_columnconfigure(0, weight=1)
        
        # Create text input box - increased height to match buttons
        self.text_input = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=10)
        self.text_input.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Create button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=1, sticky="ens", padx=5, pady=5)
        
        # Create send button
        self.send_button = ttk.Button(button_frame, text="Send", command=self.send_message)
        self.send_button.grid(row=0, column=0, padx=5, pady=5)
        
        # Create commands frame - removed the "Commands" label
        commands_frame = ttk.Frame(button_frame)
        commands_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        self.reset_button = ttk.Button(commands_frame, text="Reset", command=self.reset_chat)
        self.reset_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.snapshot_button = ttk.Button(commands_frame, text="Snapshot", command=self.take_snapshot)
        self.snapshot_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        self.goback_button = ttk.Button(commands_frame, text="Go Back", command=self.go_back)
        self.goback_button.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        self.load_history_button = ttk.Button(commands_frame, text="Load History", command=self.load_history)
        self.load_history_button.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=1, column=0, sticky="ew")
        
        # Bind Enter key to send message
        self.text_input.bind("<Control-Return>", self.send_message)
        
        # Initialize backend components
        self.initialize_backend()
        
    def initialize_backend(self):
        """Initialize chat components"""
        # Load API keys
        self.api_keys = load_api_keys()
        
        # Initialize chat history manager
        self.chatman = ChatHistoryManager(self.args.model_name, self.args.output_dir, self.args.nick_name)
        
        # Load history if specified
        if self.args.on_history_file and os.path.exists(self.args.on_history_file):
            self.append_to_conversation(f"Loading chat history from {self.args.on_history_file}", "info")
            success = self.chatman.load_history(self.args.on_history_file)
            if success:
                self.append_to_conversation(f"Loaded history with {self.chatman.session_count} previous sessions", "info")
                # Display loaded history
                for msg in self.chatman.history:
                    if msg.get("role") == self.chatman.role_legal_name_dict["user"]:
                        self.append_to_conversation(f"You: {msg.get('content', '')}", "user")
                    elif msg.get("role") == self.chatman.role_legal_name_dict["model"]:
                        self.append_to_conversation(f"{self.args.model_name.replace('-', ' ').title()}: {msg.get('content', '')}", "model")
            else:
                self.append_to_conversation("Failed to load history, starting fresh", "error")
        
        # Initialize client
        self.client = get_client(self.args.model_name, self.api_keys, self.args.temperature, self.args.max_output_tokens)
        
        # For Gemini models, just initialize message template for future use
        if self.args.model_name.startswith("gemini"):
            # Create a temporary chat session to get message structure template
            temp_chat = self.client.start_chat()
            response = temp_chat.send_message("hello")
            
            # Save the message template
            if hasattr(temp_chat, 'history') and len(temp_chat.history) > 0:
                self.client._first_message_template = temp_chat.history[0]
                self.append_to_conversation("Initialized Gemini message template", "info")
            else:
                self.append_to_conversation("Warning: Could not initialize Gemini message template", "error")
        
        # Initialize token counter
        self.token_counter = TokenCounter(self.args.model_name, self.client if self.args.model_name.startswith("gemini") else None)
        
        # Save initial meta info
        meta_file = save_meta_info(self.chatman, self.token_counter, self.args)
        self.append_to_conversation(f"Meta info saved to {meta_file}", "info")
        
        # Welcome message
        model_name_display = self.args.model_name.replace("-", " ").title()
        self.append_to_conversation(f"=== Chat with {model_name_display} ===", "info")
        self.append_to_conversation("Type your message and press Ctrl+Enter or click Send to chat", "info")
    
    def append_to_conversation(self, text, tag=None):
        """Add text to the conversation display with optional tag for color"""
        self.conversation_display.config(state=tk.NORMAL)
        self.conversation_display.insert(tk.END, text + "\n", tag)
        self.conversation_display.see(tk.END)  # Scroll to bottom
        self.conversation_display.config(state=tk.DISABLED)
    
    def send_message(self, event=None):
        """Send the message from the text input box"""
        user_input = self.text_input.get("1.0", tk.END).strip()
        
        # Check if input is empty
        if not user_input:
            self.update_status("Input is empty. Please type a message.")
            return
        
        # Clear input box
        self.text_input.delete("1.0", tk.END)
        
        # Display user message
        self.append_to_conversation(f"You: {user_input}", "user")
        
        # Count tokens in user input
        input_tokens = self.token_counter.add_input_tokens(user_input)
        self.append_to_conversation(f"[Input tokens: {input_tokens}]", "info")
        
        # Add user message to history
        self.chatman.add_user_message(user_input)
        
        # Update UI
        self.update_status("Waiting for model response...")
        self.root.update()
        
        # Call API for response
        model_family = judge_model_family(self.args.model_name)
        model_name_display = self.args.model_name.replace("-", " ").title()
        
        start_time = time.time()
        
        response = call_api(self.client, self.args.model_name, model_family, self.chatman.history, 
                        self.args.max_output_tokens, self.args.temperature)
        response_time = time.time() - start_time
        
        # Display response
        self.append_to_conversation(f"{model_name_display}: {response}", "model")
        
        # Count tokens in response
        output_tokens = self.token_counter.add_output_tokens(response)
        session_info = f"[Round {self.chatman.session_count}]"
        output_info = f"[Output tokens: {output_tokens} | Response time: {response_time:.2f}s]"
        total_info = f"[Total: Input tokens: {self.token_counter.total_input_tokens}, Output tokens: {self.token_counter.total_output_tokens}]"
        
        self.append_to_conversation(session_info, "info")
        self.append_to_conversation(output_info, "info")
        self.append_to_conversation(total_info, "info")
        
        # Add model response to history
        self.chatman.add_model_message(response)
        
        # Save history if auto-save is enabled
        if self.args.autosave_history:
            history_file = self.chatman.save_history(export_format=self.args.export_format)
            meta_file = save_meta_info(self.chatman, self.token_counter, self.args)
            self.append_to_conversation("History and meta saved", "info")
        
        # Update status
        self.update_status("Ready")
    
    def reset_chat(self):
        """Reset the chat history"""
        self.chatman.reset()
        self.token_counter.reset()
        self.append_to_conversation("Chat history reset. Starting fresh conversation.", "info")
        meta_file = save_meta_info(self.chatman, self.token_counter, self.args)
        self.append_to_conversation(f"New meta info saved to {meta_file}", "info")
    
    def take_snapshot(self):
        """Take a snapshot of the current chat"""
        history_file = self.chatman.save_history(is_snapshot=True, export_format=self.args.export_format)
        meta_file = save_meta_info(self.chatman, self.token_counter, self.args, is_snapshot=True)
        self.append_to_conversation(f"Snapshot saved to {history_file} and {meta_file}", "info")
    
    def go_back(self):
        """Remove the last exchange"""
        if len(self.chatman.history) >= 2:
            # Get the messages to be removed for token counting
            model_message = self.chatman.history[-1]
            user_message = self.chatman.history[-2]
            
            # Subtract tokens before removing the messages
            if model_message.get("role") == self.chatman.role_legal_name_dict["model"]:
                self.token_counter.subtract_output_tokens(model_message.get("content", ""))
            if user_message.get("role") == self.chatman.role_legal_name_dict["user"]:
                self.token_counter.subtract_input_tokens(user_message.get("content", ""))
            
        # Now remove the last exchange
        success = self.chatman.remove_last_exchange()
        if success:
            self.append_to_conversation("Removed last exchange.", "info")
            # Save updated history if auto-save is enabled
            if self.args.autosave_history:
                history_file = self.chatman.save_history(export_format=self.args.export_format)
                meta_file = save_meta_info(self.chatman, self.token_counter, self.args)
                self.append_to_conversation(f"Updated history saved to {history_file} and {meta_file}", "info")

            # Print the updated round and total input output tokens
            self.append_to_conversation(f"[Round {self.chatman.session_count}]", "info")
            self.append_to_conversation(f"[Total: Input tokens: {self.token_counter.total_input_tokens}, Output tokens: {self.token_counter.total_output_tokens}]", "info")
            
            # Clear the conversation display and repopulate it with the current history
            self.conversation_display.config(state=tk.NORMAL)
            self.conversation_display.delete("1.0", tk.END)
            self.conversation_display.config(state=tk.DISABLED)
            
            # Display current history
            for msg in self.chatman.history:
                if msg.get("role") == self.chatman.role_legal_name_dict["user"]:
                    self.append_to_conversation(f"You: {msg.get('content', '')}", "user")
                elif msg.get("role") == self.chatman.role_legal_name_dict["model"]:
                    self.append_to_conversation(f"{self.args.model_name.replace('-', ' ').title()}: {msg.get('content', '')}", "model")
        else:
            self.append_to_conversation("No exchanges to remove.", "error")
    
    def load_history(self):
        """Load chat history from a file"""
        file_path = filedialog.askopenfilename(
            title="Select History File",
            filetypes=(("Pickle files", "*.pkl"), ("JSON files", "*.json"), ("All files", "*.*")),
            initialdir=self.args.output_dir
        )
        
        if file_path:
            self.append_to_conversation(f"Loading chat history from {file_path}", "info")
            success = self.chatman.load_history(file_path)
            if success:
                # Clear the conversation display
                self.conversation_display.config(state=tk.NORMAL)
                self.conversation_display.delete("1.0", tk.END)
                self.conversation_display.config(state=tk.DISABLED)
                
                # Welcome message
                model_name_display = self.args.model_name.replace("-", " ").title()
                self.append_to_conversation(f"=== Chat with {model_name_display} ===", "info")
                
                # Show loaded history
                self.append_to_conversation(f"Loaded history with {self.chatman.session_count} previous sessions", "info")
                for msg in self.chatman.history:
                    if msg.get("role") == self.chatman.role_legal_name_dict["user"]:
                        self.append_to_conversation(f"You: {msg.get('content', '')}", "user")
                    elif msg.get("role") == self.chatman.role_legal_name_dict["model"]:
                        self.append_to_conversation(f"{self.args.model_name.replace('-', ' ').title()}: {msg.get('content', '')}", "model")
                
                # Reset token counter
                self.token_counter.reset()
                
                # Update status
                self.update_status(f"Loaded history from {os.path.basename(file_path)}")
            else:
                self.append_to_conversation("Failed to load history", "error")
    
    def update_status(self, message):
        """Update the status bar message"""
        self.status_bar.config(text=message)
        self.root.update()
    
    def run(self):
        """Run the GUI main loop"""
        self.root.mainloop()

def main():
    parser = argparse.ArgumentParser(description="Chat Terminal with LLMs")
    parser.add_argument("--model_name", "-m", type=str, default="gpt-4o",
                        help="Model to chat with (gpt-*, deepseek-*, gemini-*, qwen-*, llama-*, nvidia/*)")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                        help="Temperature for response generation")
    parser.add_argument("--output_dir", "-o", type=str, default="chat_logs",
                        help="Directory to save chat history and meta files")
    parser.add_argument("--nick_name", "-n", type=str, default="chat",
                        help="Nickname for the chat session (used in filenames)")
    parser.add_argument("--max_output_tokens", type=int, default=2000,
                        help="Maximum number of tokens in model responses")
    parser.add_argument("--autosave_history", type=int, default=0,
                        help="Automatically save chat history after each exchange")
    parser.add_argument("--on_history_file", type=str, default=None,
                        help="Path to a previous chat history file to continue from")
    parser.add_argument("--export_format", type=str, default="pickle", choices=["json", "pickle"],
                        help="Format to export chat history (json or pickle)")
    
    args = parser.parse_args()

    ## fix max token for deepseek
    if args.model_name.startswith("deepseek"):
        args.max_output_tokens = 8192
    elif args.model_name.startswith("gemini-2.5-pro") or args.model_name.startswith("gemini-2.0-flash-thinking"):
        args.max_output_tokens = None
    elif args.model_name.startswith("qwen"):
    ## Set appropriate max tokens for Qwen models if not specified
    # elif args.model_name.startswith("qwen") and args.max_output_tokens == 2000:
        args.max_output_tokens = 4096
    
    # Create and run the GUI
    app = ChatGUI(args)
    app.run()

if __name__ == "__main__":
    main() 