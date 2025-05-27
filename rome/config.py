# config.py
import yaml
import os
import sys
from typing import Dict, Any
from .logger import get_logger

# How long to display long strings for console output
SUMMARY_LENGTH = 100

# Meta directory for file specific information
META_DIR_EXT = 'rome'

# Default logging base directory name for agents
LOG_DIR_NAME = "__rome__"

# Define the default configuration structure as a dictionary
DEFAULT_CONFIG = {
    # OpenAIHandler settings - includes all OpenAI API configuration
    "OpenAIHandler": {
        # OpenAI API configuration
        "base_url": "https://api.openai.com/v1",
        "key_name": "OPENAI_API_KEY",
        "timeout": 30,

        # General LLM parameters
        "model": "gpt-4o",
        "temperature": 0.1,
        "max_tokens": 8192,
        "top_p": 1.0,
        "seed": None,

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
        "history_context_len": 10, # Length of history to use when selecting action
    },

    "AgentApi": {
        "host": "localhost",
        "port": 8000
    },

    # Logging configuration
    "Logger": {
        "level": "ERROR",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "console": True,
        "include_caller_info": "rome",
        "base_dir": None,  # Directory for log files (Agent sets it if None)
        "filename": None  # Log file name (Agent sets it if None)
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
        "selection_criteria": None,
        "batch_size": 5,
        # "epsilon_oldest": 0.0, # Prob to choose the oldest file for editing
        # 'exploration_strategy': 'adaptive',  # or 'breadth_first', 'depth_first', 'novelty_seeking'
        # 'diversity_weight': 1.5,
        # 'novelty_bonus': 2.0,
        # 'dependency_analysis': True,
        # 'semantic_clustering': True,
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
        "timeout": 10,
        "virtual_env_context": None,
        "work_dir": "./",
        "cmd_args": {"pytest": "-vvs"}
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
        logger.debug(f"'{attr}' provided in {obj.__class__.__name__} config")


def check_opt_attrs(obj, optional_attrs):
    """Helper function to check if optional attributes have been set"""
    logger = get_logger()
    for attr in optional_attrs:
        if not hasattr(obj, attr):
            setattr(obj, attr, None)
            logger.debug(f"'{attr}' (optional) not provided in {obj.__class__.__name__} config")


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
    import copy
    logger = get_logger()

    merged_config = copy.deepcopy(DEFAULT_CONFIG)

    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_dict(d[k], v)
            else:
                d[k] = v
        return d

    result = update_dict(merged_config, custom_config)
    logger.info("Configuration merged with defaults")
    return result


if __name__ == "__main__":
    pass
