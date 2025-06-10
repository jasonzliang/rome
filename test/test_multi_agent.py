# RUN MANUALLY ONLY
import os
import pprint
import sys
import shutil
import json
from pathlib import Path
import traceback

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rome.multi_agent import MultiAgent
from rome.logger import get_logger

# HumanEval samples for testing (4 problems)
HUMAN_EVAL_SAMPLES = {
    "HumanEval_0": '''def has_close_elements(numbers, threshold):
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    pass''',

    "HumanEval_1": '''def separate_paren_groups(paren_string):
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    pass''',

    "HumanEval_2": '''def truncate_number(number):
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
    pass''',
}

# Simplified agent configurations (2 agents)
AGENT_CONFIGS = {
    "MathSolver": "You are an expert mathematician that specializes in implementing mathematical algorithms and numerical computations with high precision.",
    "StringProcessor": "You are a string processing expert that excels at parsing, manipulating, and analyzing text and string patterns."
}


def setup_test_environment():
    """Setup test directory with HumanEval samples and agent config"""
    # Create test directory
    results_dir = Path("result")
    results_dir.mkdir(exist_ok=True)

    test_dir = results_dir / "test_multi_agent"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    # Create problem directories and files
    for problem_name, content in HUMAN_EVAL_SAMPLES.items():
        problem_dir = test_dir / problem_name
        problem_dir.mkdir()
        (problem_dir / f"{problem_name}.py").write_text(content)

    # Create agent role JSON
    config_file = test_dir / "agent_roles.json"
    config_file.write_text(json.dumps(AGENT_CONFIGS, indent=4))

    return test_dir, str(config_file)


def create_config():
    """Create compact configuration for multi-agent system"""
    return {
        "OpenAIHandler": {
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 8192
        },
        "Agent": {
            "fsm_type": "simple",
            "patience": 1,
            "agent_api": False
        },
        "Logger": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "console": True,
            "include_caller_info": "rome"
        },
        "TournamentSearchAction": {
            "batch_size": 2
        },
        "PrioritySearchAction": {
            "batch_size": 2
        },
        "RepositoryManager": {
            "file_types": [".py"]
        },
    }


def main():
    """Main execution function"""
    # Setup logging
    logger = get_logger()
    logger.configure({
        "level": "DEBUG",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "console": True,
        "include_caller_info": "rome"
    })

    logger.info("Starting multi-agent test")

    # Setup environment
    test_dir, agent_role_json = setup_test_environment()
    logger.info(f"Test environment created: {test_dir}")

    # Create multi-agent system
    config = create_config()

    multi_agent = MultiAgent(
        agent_role_json=agent_role_json,
        repository=str(test_dir.absolute()),
        config=config
    )

    # Run with user input
    results = None
    while True:
        try:
            iterations = int(input("Enter number of iterations (0 to quit): "))
            if iterations == 0:
                break

            logger.info(f"Running {iterations} iterations")
            results = multi_agent.run_loop(max_iterations=iterations)

        except (ValueError, KeyboardInterrupt):
            logger.info("Exiting")
            break

    # Cleanup and save results
    multi_agent.shutdown()

    if not results:
        results = {
            'total_agents': len(AGENT_CONFIGS),
            'completed_agents': 0,
            'agent_results': {},
            'repository': str(test_dir.absolute()),
            'log_dir': multi_agent.get_log_dir()
        }

    # Log summary
    logger.info("=" * 80)
    logger.info("EXECUTION COMPLETED")
    logger.info("=" * 80)

    summary = results.get('summary', {})
    if summary:
       for line in pprint.pformat(summary, width=80).split('\n'):
           logger.info(line)
    else:
       logger.info("No summary data available")

    # Save results
    log_dir = Path(results['log_dir'])
    try:
        # Save main results
        with open(log_dir / "multi_agent_results.json", "w") as f:
            json.dump(results, f, indent=4, default=str)
        logger.info(f"Results saved to: {log_dir}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
