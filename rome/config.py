# default_config.py
import yaml
import os

# Define the default configuration structure as a dictionary
DEFAULT_CONFIG = {
    # LLM settings - includes all OpenAI API configuration
    "llm": {
        # OpenAI API configuration
        "api_key": None,  # Set to None, will use environment variable if not provided
        "api_base": "https://api.openai.com/v1",
        "api_type": None,  # "openai" (default) or "azure" or other supported providers
        "api_version": None,  # Required for Azure OpenAI
        "timeout": 120,
        "use_cache": False,

        # General LLM parameters
        "model": "gpt-4",
        "temperature": 0.2,
        "max_tokens": 4000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,

        # Caching settings
        "cache_enabled": False,
        "cache_seed": 42,

        # System message for chat completions
        "system_message": "You are a helpful code assistant specializing in code analysis and improvement."
    },

    # Repository settings
    "repository": {
        "path": "./",
        "extensions": [".py", ".js", ".ts", ".java", ".rb", ".go", ".c", ".cpp", ".h", ".hpp"]
    },

    # Logging configuration
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": None,  # Set to a path to enable file logging
        "console": True
    },

    # Safety settings
    "safety": {
        "create_backup": True,
        "backup_dir": "./backups",
        "preview_changes": True,
        "max_file_size": 5000000  # 5MB
    },

    # FSM configuration
    "fsm": {
        "initial_state": "IDLE",
        "strict_transitions": False,
        "default_error_state": "ERROR"
    },

    # Action-specific configurations
    "actions": {
        "search": {
            "max_files": 100,
            "include_content": True,
            "depth": 3,
            "exclude_dirs": [".git", "node_modules", "venv", "__pycache__", "dist", "build"],
            # Example LLM override for search action
            "llm": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.1,
                "max_tokens": 1000,
                "system_message": "You are a search assistant that helps find and analyze code files."
            }
        }
    },
}

def generate_default_config(output_path="config.yaml"):
    """Generate a default configuration YAML file"""
    with open(output_path, 'w') as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
    print(f"Default configuration saved to {output_path}")

def load_config(config_path="config.yaml", create_if_missing=True):
    """Load configuration from a YAML file, creating it if it doesn't exist"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    elif create_if_missing:
        print(f"Config file {config_path} not found. Creating default config...")
        generate_default_config(config_path)
        return DEFAULT_CONFIG
    else:
        print(f"Config file {config_path} not found.")
        return DEFAULT_CONFIG

def merge_with_default_config(custom_config):
    """Merge a custom config with the default config"""
    import copy
    merged_config = copy.deepcopy(DEFAULT_CONFIG)

    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_dict(d[k], v)
            else:
                d[k] = v
        return d

    return update_dict(merged_config, custom_config)

def get_action_llm_config(config, action_name):
    """
    Get the LLM configuration for a specific action, merging with the main LLM config.

    Args:
        config: The full configuration dictionary
        action_name: Name of the action (e.g., 'search')

    Returns:
        Dict: LLM configuration with action-specific overrides applied
    """
    import copy

    # Start with the main LLM config
    base_llm_config = copy.deepcopy(config.get('llm', {}))

    # Get action-specific config
    action_config = config.get('actions', {}).get(action_name, {})
    action_llm_config = action_config.get('llm', {})

    # Merge action-specific LLM config over the base config
    if action_llm_config:
        base_llm_config.update(action_llm_config)

    return base_llm_config

if __name__ == "__main__":
    # Generate default config if script is run directly
    generate_default_config()