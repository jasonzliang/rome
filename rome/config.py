# default_config.py
import yaml
import os

# Define the default configuration structure as a dictionary
DEFAULT_CONFIG = {
    # OpenAI API configuration
    "openai": {
        "api_key": None,  # Set to None, will use environment variable if not provided
        "api_base": "https://api.openai.com/v1",
        "timeout": 120
    },

    # LLM settings
    "llm": {
        "model": "gpt-4",
        "temperature": 0.2,
        "max_tokens": 4000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
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
            "exclude_dirs": [".git", "node_modules", "venv", "__pycache__", "dist", "build"]
        },
        "analyze": {
            "default_prompt": "Analyze the following code and provide a summary of its functionality, potential issues, and improvement suggestions:",
            "model": "gpt-4",
            "temperature": 0.2,
            "max_tokens": 2000,
            "use_context": True
        },
        "update": {
            "create_backup": True,
            "model": "gpt-4",
            "temperature": 0.1,
            "max_tokens": 4000,
            "preview_changes": True,
            "update_prompt_template": """
            You are tasked with updating the following {file_type} code according to these instructions:

            INSTRUCTIONS:
            {instructions}

            ORIGINAL CODE:
            ```
            {original_content}
            ```

            Please provide ONLY the updated code without any explanations or markdown formatting.
            """
        }
    },

    # Customization options
    "prompts": {
        "analyze": None,  # If None, default_prompt from actions.analyze will be used
        "update": None,   # If None, update_prompt_template from actions.update will be used
        "system_message": "You are a helpful code assistant specializing in code analysis and improvement."
    }
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
    # Create a deep copy of the default config
    import copy
    merged_config = copy.deepcopy(DEFAULT_CONFIG)

    # Recursive function to update nested dictionaries
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_dict(d[k], v)
            else:
                d[k] = v
        return d

    # Update the default config with custom values
    merged_config = update_dict(merged_config, custom_config)
    return merged_config

if __name__ == "__main__":
    # Generate default config if script is run directly
    generate_default_config()
