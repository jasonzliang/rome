#!/usr/bin/env python3
"""
CaesarAgent CLI Runner

Usage:
    python run_agent.py <repository_path> <config_path> [--max-iterations N]

Example:
    python run_agent.py ./my_project config.yaml
    python run_agent.py ./my_project config.yaml --max-iterations 50
"""

import argparse
import sys
import os
import pprint
from pathlib import Path
import traceback

# Add the parent directory to sys.path to import from the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecology.caesar_agent import CaesarAgent
from rome.config import load_config, format_yaml_like
from rome.logger import get_logger


def validate_repository(repo_path: str, logger) -> str:
    """Validate and return absolute repository path, creating if needed"""
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
  python run_agent.py ./my_project config.yaml --max-iterations 50
  python run_agent.py ./other_project another_config.yaml --max-iterations 100
        """
    )

    parser.add_argument('repository', type=str, help='Path to repository directory')
    parser.add_argument('config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--max-iterations', type=int, default=None,
                       help='Override max_iterations from config (default: use config value)')

    return parser.parse_args()


def print_config_summary(agent, logger):
    """Print configuration summary before starting"""
    summary = {
        "Agent Configuration": {
            "Name": agent.name,
            "Repository": agent.repository,
            "Max Iterations": agent.max_iterations,
        },
        "Caesar Settings": {
            "Starting URL": agent.starting_url,
            "Starting Query": agent.starting_query,
            "Allowed Domains": ', '.join(agent.allowed_domains) if not agent.allow_all_domains else "* (ALL DOMAINS)",
            "Max Depth": agent.max_depth,
            "Save Interval": agent.save_graph_interval,
            "Draw Graph": agent.draw_graph,
            "Exploration LLM Config": agent.exploration_llm_config,
            "Checkpoint Interval": agent.checkpoint_interval,
        },
        "OpenAI Settings": {
            "Model": agent.openai_handler.model,
            "Cost Limit": f"${agent.openai_handler.cost_limit:.2f}" if agent.openai_handler.cost_limit else "None",
        }
    }

    logger.info("="*80)
    logger.info("CAESAR AGENT CONFIGURATION")
    logger.info("="*80)
    for line in format_yaml_like(summary):
        logger.info(line)
    logger.info("="*80)


def print_final_summary(agent, artifact, logger):
    """Print exploration completion summary"""
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
    if cost.get('cost_limit'):
        remaining = cost.get('remaining_budget', 0)
        usage_pct = (cost['accumulated_cost'] / cost['cost_limit']) * 100
        logger.info(f"  Budget Used:   {usage_pct:.1f}% (${remaining:.2f} remaining)")

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

    try:
        # Parse arguments
        args = parse_args()

        # Validate repository
        repository = validate_repository(args.repository, logger)

        # Load configuration
        try:
            config = load_config(args.config)
            logger.info(f"Configuration loaded: {args.config}")
            logger.info(pprint.pformat(config))
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)

        # Override max_iterations if provided via CLI
        if args.max_iterations is not None:
            if 'CaesarAgent' not in config:
                config['CaesarAgent'] = {}
            config['CaesarAgent']['max_iterations'] = args.max_iterations
            logger.info(f"Overriding max_iterations from CLI: {args.max_iterations}")

        # Get agent name from config
        agent_name = config.get('Agent', {}).get('name', 'CaesarAgent')

        # Initialize agent
        logger.info("Initializing CaesarAgent...")
        agent = CaesarAgent(name=agent_name, repository=repository, config=config)

        # Print configuration
        print_config_summary(agent, logger)

        # Run exploration
        logger.info("Starting exploration...\n")
        artifact = agent.explore()

        # Print final summary
        print_final_summary(agent, artifact, logger)

        # Shutdown
        agent.shutdown()
        return 0

    except KeyboardInterrupt:
        logger.error("\nExploration interrupted by user")
        if 'agent' in locals():
            agent.shutdown()
        return 130

    except Exception as e:
        logger.error(f"\nError: {e}")
        traceback.print_exc()
        if 'agent' in locals():
            agent.shutdown()
        return 1


if __name__ == '__main__':
    sys.exit(main())