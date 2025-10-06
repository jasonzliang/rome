#!/usr/bin/env python3
"""
CaesarAgent CLI Runner

Usage:
    python run_agent.py <repository_path> <config_path>

Example:
    python run_agent.py ./my_project config.yaml
"""

import argparse
import sys
import os
from pathlib import Path
import traceback

# Add the parent directory to sys.path to import from the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rome.caesar_agent import CaesarAgent
from rome.config import load_config, merge_with_default_config, format_yaml_like
from rome.logger import get_logger


def validate_repository(repo_path: str) -> str:
    """Validate and return absolute repository path, creating if needed"""
    logger = get_logger()
    path = Path(repo_path).resolve()

    if not path.exists():
        logger.info(f"Creating repository directory: {path}")
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            sys.exit(1)
    elif not path.is_dir():
        logger.error(f"Path exists but is not a directory: {path}")
        sys.exit(1)

    return str(path)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run CaesarAgent for web exploration and insight synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agent.py ./my_project config.yaml
  python run_agent.py ./other_project another_config.yaml
        """
    )

    parser.add_argument('repository', type=str, help='Path to repository directory')
    parser.add_argument('config', type=str, help='Path to YAML configuration file')

    return parser.parse_args()


def load_and_merge_config(config_path: str) -> dict:
    """Load and merge configuration with defaults"""
    logger = get_logger()

    try:
        config = load_config(config_path)
        config = merge_with_default_config(config)
        logger.info(f"Configuration loaded: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)


def print_config_summary(agent):
    """Print configuration summary before starting"""
    summary = {
        "Agent Configuration": {
            "Name": agent.name,
            "Repository": agent.repository,
            "Max Iterations": agent.max_iterations,
        },
        "Caesar Settings": {
            "Starting URL": agent.starting_url,
            "Allowed Domains": ', '.join(agent.allowed_domains) if not agent.allow_all_domains else "* (ALL DOMAINS)",
            "Max Depth": agent.max_depth,
            "Save Interval": agent.save_graph_interval,
            "Draw Graph": agent.draw_graph,
            "Temperature": agent.exploration_temperature,
        },
        "OpenAI Settings": {
            "Model": agent.openai_handler.model,
            "Cost Limit": f"${agent.openai_handler.cost_limit:.2f}" if agent.openai_handler.cost_limit else "None",
        }
    }

    print("="*80)
    print("CAESAR AGENT CONFIGURATION")
    print("="*80)
    for line in format_yaml_like(summary):
        print(line)
    print("="*80)
    print()


def print_final_summary(agent, artifact):
    """Print exploration completion summary"""
    logger = get_logger()
    cost = agent.openai_handler.get_cost_summary()

    logger.info("\n" + "="*80)
    logger.info("EXPLORATION COMPLETE - FINAL SYNTHESIS")
    logger.info("="*80)
    logger.info(artifact)
    logger.info("="*80)

    logger.info("\nAPI COST SUMMARY:")
    logger.info(f"  Total Cost:    ${cost['accumulated_cost']:.4f}")
    logger.info(f"  API Calls:     {cost['call_count']}")
    logger.info(f"  Avg per Call:  ${cost['average_cost_per_call']:.4f}")
    if cost['cost_limit']:
        logger.info(f"  Budget Used:   {cost['usage_percentage']}")

    logger.info(f"\nLogs saved to: {agent.get_log_dir()}")
    logger.info("Exploration completed successfully!")


def main():
    """Main execution function"""
    # Configure logger
    logger = get_logger()
    logger.configure({
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "console": True
    })

    # Parse arguments
    args = parse_args()

    # Validate repository
    repository = validate_repository(args.repository)

    # Load configuration
    config = load_and_merge_config(args.config)

    # Get agent name from config
    agent_name = config.get('Agent', {}).get('name', 'CaesarAgent')

    # Initialize agent
    logger.info("Initializing CaesarAgent...")
    try:
        agent = CaesarAgent(name=agent_name, repository=repository, config=config)
    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Print configuration
    print_config_summary(agent)

    # Run exploration
    logger.info("Starting exploration...\n")

    try:
        artifact = agent.explore()
        print_final_summary(agent, artifact)
        agent.shutdown()
        return 0

    except KeyboardInterrupt:
        logger.error("\nExploration interrupted by user")
        agent.shutdown()
        return 130

    except Exception as e:
        logger.error(f"\nExploration failed: {e}")
        traceback.print_exc()
        agent.shutdown()
        return 1


if __name__ == '__main__':
    sys.exit(main())
