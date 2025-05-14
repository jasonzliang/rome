# example_usage.py
import os
from agent import Agent
from default_config import generate_default_config

def main():
    # Generate default config file if it doesn't exist
    if not os.path.exists('config.yaml'):
        generate_default_config('config.yaml')
        print("Generated default configuration file: config.yaml")

    # Initialize the agent with the default config
    agent = Agent(config_path='config.yaml')
    print(f"Agent initialized with state: {agent.get_current_state()}")

    # Example of overriding config with a custom dictionary
    custom_config = {
        "llm": {
            "model": "gpt-3.5-turbo",  # Use a different model
            "temperature": 0.5,
        },
        "repository": {
            "path": "./my_project"  # Custom repository path
        }
    }

    # Create a new agent with custom config (merged with defaults)
    custom_agent = Agent(config_dict=custom_config)
    print(f"Custom agent initialized with state: {custom_agent.get_current_state()}")

    # Search for files containing a specific pattern
    results = agent.search_repository(query="def process_data", file_pattern="*.py")
    print(f"Found {len(results)} files containing 'def process_data'")

    # Analyze the code
    agent.analyze_code()

    # Update a specific file if found
    if results:
        update_result = agent.update_file(
            file_path=results[0]['path'],
            instructions="Optimize the process_data function for better performance and add docstrings"
        )
        print(f"File update {'successful' if update_result.get('success') else 'failed'}")


if __name__ == "__main__":
    main()
