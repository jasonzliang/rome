# config.py
import copy
import os
import sys
import yaml
from types import NoneType
from typing import Dict, Any
from .logger import get_logger

######## These constants are not intended to be user modifiable ########
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
######## These constants are not intended to be user modifiable ########

# Define the default configuration structure as a dictionary
DEFAULT_CONFIG = {
    # OpenAIHandler settings - includes all OpenAI API configuration
    "OpenAIHandler": {
        # OpenAI API configuration
        "base_url": "https://api.openai.com/v1", # Url for chat completion API
        "key_name": "OPENAI_API_KEY", # API key name to find in env
        "timeout": 60, # Time for allowing responses from API

        # General LLM parameters
        "model": "gpt-4o", # LLM model name
        "temperature": 0.1, # LLM temperature
        "max_tokens": 8192, # LLM response max tokens
        "top_p": 1.0, # LLM top-P
        "seed": None, # LLM random seed

        # Fall back system message for chat completions if none given
        "system_message": "You are a helpful code assistant specializing in code analysis and improvement.",

        # Context management parameters
        "manage_context": True, # Prevent messages from overfilling context window
        "max_input_tokens": None, # Set to manually override auto-calculated input token limit
        "token_count_thres": 0.8, # Threshold when to use slow method to count number of tokens
        "chars_per_token": 4 # Used for fast calculation of number of tokens
    },

    # Agent configuration
    "Agent": {
        "repository": "./", # Code repository base directory
        "fsm_type": "minimal", # Which FSM to load (see fsm.py)
        "agent_api": True, # Launch an REST API server for agent's internal state
        "history_context_len": 15, # Length of history to use when selecting action
        "patience": 3, # If same state/action chosen repeatedly, prompt to choose different action
    },

    "AgentApi": {
        "host": "localhost", # Url to query Agent API
        "port": 8000 # Port to query agent API
    },

    # Logging configuration
    "Logger": {
        "level": "ERROR", # Log level in increasing verbosity: INFO -> ERROR -> DEBUG
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s", # Formatting for log messages
        "console": True, # Print to console if true
        "include_caller_info": None, # Can be "rome", "rich", or None
        "base_dir": None, # Directory for log files (overwrites agent's setting)
        "filename": None, # Log file name (overwrites agent's setting)
        "max_size_kb": 10000, # Max log size in kb, truncates after exceeding
    },

    # FSM configuration
    "FSM": {},

    # State configurations
    "IdleState": {},
    "CodeLoadedState": {},
    "CodeEditedState": {},
    "TestEditedState": {},

    # Action configuration
    "SearchAction": {
        "max_files": sys.maxsize,
        "file_types": [".py"],
        "exclude_types": ["_test.py"],
        "exclude_dirs": [".git", "venv", "__pycache__"],
        "selection_criteria": "Select the file you have the most confidence in improving or fixing.",
        "batch_size": 10,
        "batch_sampling": False
    },
    "ResetAction": {},
    "AdvancedResetAction": {},
    "EditCodeAction": {
        "custom_prompt": None
    },
    "EditTestAction": {
        "custom_prompt": None
    },
    "ExecuteCodeAction": {},
    "TransitionAction": {},

    # Code executor configuration
    "Executor": {
        "timeout": 10, # Maximum time for code to run
        "virtual_env_context": None, # Name of virtual env to run in
        "work_dir": "./", #
        "cmd_args": {"pytest": ["-vvs", "--tb=long"]}
    },

    # Version manager configuration
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


def load_config(config_path="config.yaml", create_if_missing=True):
    """Load configuration from a YAML file, creating it if it doesn't exist"""
    logger = get_logger()

    if os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    elif create_if_missing:
        logger.info(f"Config file {config_path} not found. Creating default config...")
        generate_default_config(config_path)
        return DEFAULT_CONFIG
    else:
        logger.error(f"Config file {config_path} not found.")
        raise FileNotFoundError(f"Config file {config_path} not found.")


def merge_with_default_config(custom_config):
    """Merge a custom config with the default config"""
    logger = get_logger()
    merged_config = copy.deepcopy(DEFAULT_CONFIG)

    def _validate(key, orig_v, new_v):
        def _error_msg(key, val):
            return f"Invalid value {val} ({type(val).__name__}) for config parameter '{key}'"

        if orig_v is not None:
            logger.assert_true(type(orig_v) == type(new_v), f"Type mismatch for old {type(orig_v).__name__} and new value {type(orig_v).__name__} for config parameter '{key}'")
        else:
            logger.debug(f"Type matching skipped for config parameter '{key}' ({orig_v} -> {new_v})")

        for v in [orig_v, new_v]:
            if isinstance(v, (list, tuple, str)):
                if not key.startswith("exclude_"):
                    logger.assert_true(len(v) > 0, f"{_error_msg(key, v)} — non-empty value")
            elif isinstance(v, (int, float)):
                logger.assert_true(v >= 0, f"{_error_msg(key, v)} — non-negative value")
            else:
                logger.assert_true(isinstance(v, (bool, NoneType)),
                    f"{_error_msg(key, v)} - invalid type")

    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_dict(d[k], v)
            else:
                _validate(k, d[k], v)
                d[k] = v
        return d

    result = update_dict(merged_config, custom_config)
    logger.info("Configuration merged with defaults")
    return result


if __name__ == "__main__":
    pass
