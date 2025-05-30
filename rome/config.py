# config.py
import ast
import copy
import os
import sys
import yaml
from types import NoneType
from typing import Dict, Any
from .logger import get_logger

######## These constants are not intended to be user modifiable ########
# Min and max allowed lengths for agent names
AGENT_NAME_LENGTH = (8, 24)

# Default hash function to use for check file versions and AST cache
DEFAULT_HASH_FUNC = "sha256"

# How long to display long strings for console output
SUMMARY_LENGTH = 100
LONG_SUMMARY_LEN = 200
LONGER_SUMMARY_LEN = 400
LONGEST_SUMMARY_LEN = 800

# Meta directory for file specific information
META_DIR_EXT = 'rome'

# Default logging base directory name for agents
LOG_DIR_NAME = "__rome__"

# Default extension for test files:
TEST_FILE_EXT = '_test.py'

######## These constants are not intended to be user modifiable ########

# Define the default configuration structure as a dictionary
# This should show the complete configuration of every class
DEFAULT_CONFIG = {
    # OpenAIHandler settings - includes all OpenAI API configuration
    "OpenAIHandler": {
        # OpenAI API configuration
        "base_url": "https://api.openai.com/v1", # Url for chat completion API
        "key_name": "OPENAI_API_KEY", # API key name to find in env
        "timeout": 60, # Time for allowing responses from API
        "cost_limit": 500, # Max budget ($) for running OpenAI chat completion

        # General LLM parameters
        "model": "gpt-4o", # LLM model name
        "temperature": 0.1, # LLM temperature
        "max_tokens": 8192, # LLM response max tokens
        "top_p": 1.0, # LLM top-P
        "seed": None, # LLM random seed

        # Fall back system message for chat completions if none given
        "system_message": "You are a helpful assistant",

        # Context management parameters
        "manage_context": True, # Prevent messages from overfilling context window
        "max_input_tokens": None, # Set to manually override auto-calculated input token limit
        "token_count_thres": 0.8, # Threshold when to use slow method to count number of tokens
        "chars_per_token": 4 # Used for fast calculation of number of tokens
    },

    # Agent configuration
    "Agent": {
        "name": None, # Agent name, can be overwritten by constructor
        "role": None, # Agent description, can be overwritten by constructor
        "repository": None, #  Repository base directory, can be overwritten by constructor
        "fsm_type": "simple", # Which FSM to load (see fsm_factory.py)
        "agent_api": True, # Launch an REST API server for agent's internal state
        "history_context_len": 15, # Length of history to use when selecting action
        "patience": 3, # If same state/action chosen repeatedly, prompt to choose different action
    },

    "AgentApi": {
        "host": "localhost", # Url/hostname to query agent API
        "port": 8000 # Port number to query agent API
    },

    # Logging configuration
    "Logger": {
        "level": "ERROR", # Log level in increasing verbosity: INFO -> ERROR -> DEBUG
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s", # Formatting for log messages
        "console": True, # Print to console if true
        "include_caller_info": None, # Can be "rome", "rich", or None
        "base_dir": None, # Directory for log files, overwrites agent's auto-generated values
        "filename": None, # Log file name, overwrites agent's auto-generated values
        "max_size_kb": 10000, # Max log size in kb, truncates after exceeding
    },

    # FSM configuration
    "FSM": {},

    # State configurations
    "IdleState": {},
    "CodeLoadedState": {},
    "CodeEditedState": {},
    "TestEditedState": {},
    "CodeExecutedPassState": {},
    "CodeExecutedFailState": {},

    # Action configuration
    "SearchAction": {
        "max_files": sys.maxsize, # Maximum number of candidates to select from after search
        "file_types": [".py"], #  File types to search for
        "exclude_types": [], # File types to not search for
        "exclude_dirs": [".*", "venv", "__*__"], # Directories to not search in
        "selection_criteria": "Select the file that you have the most confidence in improving or fixing.", # Criteria for selecting file from possible candidates
        "batch_size": 10, # Candidate batch size for selection
        "batch_sampling": False # If set to true, randomly sample batches
    },
    "ResetAction": {},
    "AdvancedResetAction": {},
    "EditCodeAction": {
        "custom_prompt": None # Override prompt for editing code
    },
    "EditTestAction": {
        "custom_prompt": None # Override prompt for editing tests
    },
    "ExecuteCodeAction": {},
    "TransitionAction": {},

    # Code executor configuration
    "Executor": {
        "timeout": 10, # Maximum time for code to run
        "virtual_env_context": None, # Name of virtual env to run in
        "work_dir": "./", # Working directory when running code directly (not code file)
        "cmd_args": { # Additional useful flags when executing code
            "pytest": ["-vvs", "--tb=long", "--no-header"],
            "python": ["-u"],
            }
    },

    # Database and version manager configuration
    "DatabaseManager": {
        "lock_timeout": 5.0, # Timeout for read/write lock
        "max_retries": 5, # Num tries to acquire lock
        "retry_delay": 0.1 # Delay between tries, exponential increase
    },
    "VersionManager": {},
}


### Helper methods for setting/loading config parameters ###
def check_attrs(obj, required_attrs):
    """Helper function to check if required attributes have been set"""
    logger = get_logger()
    for attr in required_attrs:
        logger.assert_attribute(obj, attr)
        # logger.debug(f"'{attr}' provided in {obj.__class__.__name__} config")


def check_opt_attrs(obj, optional_attrs):
    """Helper function to check if optional attributes have been set"""
    logger = get_logger()
    for attr in optional_attrs:
        if not hasattr(obj, attr):
            setattr(obj, attr, None)
            # logger.debug(f"'{attr}' (optional) not provided in {obj.__class__.__name__} config")


def set_attributes_from_config(obj, config=None, required_attrs=None, optional_attrs=None):
    """
    Helper function to convert configuration dictionary entries to object attributes

    Args:
        obj: The object to set attributes on
        config: The configuration dictionary
        required_attrs: List of attributes that must be set from config
        optional_attrs: List of attributes that are optional
    """
    logger = get_logger()

    # Set each config parameter as an attribute
    if config:
        for key, value in config.items():
            setattr(obj, key, value)

    if required_attrs:
        check_attrs(obj, required_attrs)

    if optional_attrs:
        check_opt_attrs(obj, optional_attrs)

    logger.debug(f"Applied configuration to {obj.__class__.__name__} attributes")


def generate_default_config(output_path="config.yaml"):
    """Generate a default configuration YAML file"""
    logger = get_logger()

    with open(output_path, 'w') as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Default configuration saved to {output_path}")
    print(f"Default configuration saved to {output_path}")


def load_config(config_path="config.yaml", create_if_missing=False):
    """Load configuration from a YAML file or Python file, creating it if it doesn't exist"""
    logger = get_logger()

    if os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")
        _, ext = os.path.splitext(config_path.lower())

        if ext == '.py':
            with open(config_path, 'r') as f:
                tree = ast.parse(f.read())

            # Check for first dictionary that has META_DIR_EXT (rome) in its name
            for node in tree.body:
                if (isinstance(node, ast.Assign) and
                    any(isinstance(target, ast.Name) and META_DIR_EXT.lower() in target.id.lower()
                        for target in node.targets)):
                    config = ast.literal_eval(node.value)
                    if not isinstance(config, dict):
                        raise TypeError(f"Config must be a dictionary in {config_path}")
                    return config

            raise ValueError(f"No ROME_CONFIG found in {config_path}")

        elif ext in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")

    elif create_if_missing:
        logger.info(f"Config file {config_path} not found. Creating default config...")
        generate_default_config(config_path)
        return DEFAULT_CONFIG
    else:
        raise FileNotFoundError(f"Config file {config_path} not found.")


def merge_with_default_config(custom_config):
    """Merge a custom config with the default config and do some basic validation"""

    def _error_msg(key, val):
        return f"Invalid value {val} ({type(val).__name__}) for config parameter '{key}'"

    def _validate(key, old_v, new_v):
        # Easy to way in include key but still use default
        if new_v == "default":
            new_v = old_v

        # Check old and new value type matches
        if old_v is not None:
            logger.assert_true(type(old_v) == type(new_v), f"Type mismatch for old {type(old_v).__name__} and new value {type(new_v).__name__} for config parameter '{key}'")
        else:
            logger.debug(f"Type matching skipped for config parameter '{key}' ({old_v} -> {new_v})")

        # Key value must be >= 0 or length >= 0 (exception for 'exclude_' keys)
        for v in [old_v, new_v]:
            if isinstance(v, (list, tuple, str)):
                if not key.startswith("exclude_"):
                    logger.assert_true(len(v) > 0, f"{_error_msg(key, v)} — non-empty value")
            elif isinstance(v, (int, float)):
                logger.assert_true(v >= 0, f"{_error_msg(key, v)} — non-negative value")
            else:
                logger.assert_true(isinstance(v, (bool, NoneType)),
                    f"{_error_msg(key, v)} - invalid type")
        return new_v

    def _update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = _update_dict(d[k], v)
            else:
                d[k] = _validate(k, d[k], v)
        return d

    logger = get_logger()
    merged_config = copy.deepcopy(DEFAULT_CONFIG)
    result = _update_dict(merged_config, custom_config)
    logger.info("Configuration merged with defaults")
    return result


if __name__ == "__main__":
    pass
