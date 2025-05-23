# config.py
import yaml
import os
import sys
from typing import Dict, Any
from .logger import get_logger

# Default logging base directory name for agents
DEFAULT_LOGDIR_NAME = "__rome__"

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
        "max_tokens": 4096,
        "top_p": 1.0,

        # Fall back system message for chat completions
        "system_message": "You are a helpful code assistant specializing in code analysis and improvement."
    },

    # Agent configuration
    "Agent": {
        "repository": "./", # Code repository base directory
        "fsm_type": "minimal", # Which FSM to load (see fsm.py)
        "agent_api": True # Launch an REST API server for agent's internal state
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
        "exclude_types": ["_test.py", ".orig.py"],
        "exclude_dirs": [".git", "venv", "__pycache__", DEFAULT_LOGDIR_NAME],
        "selection_criteria": None,
        "batch_size": 5,
        "epilson_oldest": 0.0 # Prob to choose the oldest file for editing
    },
    "RetryAction": {},
    "EditCodeAction": {
        "custom_prompt": None
    },
    "EditTestAction": {
        "custom_prompt": None
    },
    "ExecuteCodeAction": {
        # Code executor configuration
        "executor_config": {
            "timeout": 10,
            "virtual_env_context": None,
            "work_dir": "./",
            "cmd_args": {"pytest": "-vvs"}
        }
    }
}


def check_attrs(obj, required_attrs):
    """Helper function to check if required attributes have been set"""
    logger = get_logger()
    for attr in required_attrs:
        logger.assert_attribute(obj, attr)


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
    # Generate default config if script is run directly
    generate_default_config()