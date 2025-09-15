# RUN MANUALLY ONLY
import os
import pprint
import sys
import shutil
import json
from pathlib import Path
import traceback

# Add the parent directory to sys.path to import from the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rome.agent import Agent
from rome.logger import get_logger

# Compact set of HumanEval problems for focused testing
HUMAN_EVAL_SAMPLES = {
    "HumanEval_0": """def has_close_elements(numbers, threshold):
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
    pass""",
    "HumanEval_1": """def separate_paren_groups(paren_string):
    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    \"\"\"
    pass""",
    "HumanEval_2": """def truncate_number(number):
    \"\"\" Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    \"\"\"
    pass"""
}
# HUMAN_EVAL_SAMPLES = {}

def setup_test_dir():
    """Create test directory with HumanEval samples"""
    test_dir = Path("result/test_single_agent")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    for problem_name, content in HUMAN_EVAL_SAMPLES.items():
        problem_dir = test_dir / problem_name
        problem_dir.mkdir()
        (problem_dir / f"{problem_name}.py").write_text(content)

    return test_dir


def create_config():
    """Compact, efficient configuration"""
    return {
        "OpenAIHandler": {
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 8192,
            "cost_limit": 10.0,
        },
        "Agent": {
            "fsm_type": "knowledge_base",
            "action_select_strat": "original",
            "patience": 1,
            "agent_api": True,
            "save_hist_interval": 1
        },
        "Logger": {
            "level": "DEBUG",
            "console": True,
            "include_caller_info": "rome"
        },
        "PrioritySearchAction": {
            "batch_size": 2
        },
        "TournamentSearchAction": {
            "batch_size": 2
        },
        "RepositoryManager": {
            "file_types": [".py"]
        },
        "AdvancedResetAction": {
            "use_ground_truth": False,
            "completion_confidence": 80,
            "max_versions": 30,
        },
        "SaveKBAction": {
            "use_ground_truth": False,
            "completion_confidence": 85,
            "max_versions": 30,
        },
    }


def run_test():
    """Main test execution with minimal interaction"""
    logger = get_logger()
    logger.configure({
        "level": "DEBUG",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "console": True,
        "include_caller_info": "rome"
    })
    logger.info("Starting single-agent test")

    agent = None
    try:
        # Setup
        test_dir = setup_test_dir()
        logger.info(f"Test directory: {test_dir}")

        # Create agent
        agent = Agent(
            name="CodeExpert",
            role="You are an expert code expert that can analyze and write interesting algorithms and functions",
            repository=str(test_dir.absolute()),
            config=create_config()
        )

        # Interactive execution
        while True:
            try:
                iterations = int(input(f"\nIterations to run (0=quit): "))
                if iterations <= 0:
                    break

                results = agent.run_loop(max_iterations=iterations, stop_on_error=True)

                summary = results.get('summary', {})
                if summary:
                   for line in pprint.pformat(summary, width=80).split('\n'):
                       logger.info(line)
                else:
                   logger.info("No summary data available")

            except (ValueError, KeyboardInterrupt):
                break

        # Final results
        final_results = agent.get_summary()

        # Save results
        log_dir = Path(agent.get_log_dir())
        (log_dir / "final_results.json").write_text(json.dumps(final_results, indent=2, default=str))

        logger.info(f"Test completed. Results in: {log_dir}")
        return final_results

    except Exception as e:
        logger.error(f"Test error: {e}")
        logger.error(traceback.format_exc())
        return None
    finally:
        if agent:
            agent.shutdown()


if __name__ == "__main__":
    run_test()
