import os
import sys
import shutil
import json
from pathlib import Path

# Add the parent directory to sys.path to import from the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rome.agent import Agent
from rome.logger import get_logger

# Sample HumanEval problems to use for testing
HUMAN_EVAL_SAMPLES = {
    "HumanEval_0.py": """
def has_close_elements(numbers, threshold):
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
    pass
""",
    "HumanEval_1.py": """
def separate_paren_groups(paren_string):
    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    \"\"\"
    pass
""",
    "HumanEval_2.py": """
def truncate_number(number):
    \"\"\" Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    \"\"\"
    pass
"""
}

def setup_test_dir():
    """Create a test directory and populate it with HumanEval samples"""
    # Create results directory if it doesn't exist
    results_dir = Path("result")
    results_dir.mkdir(exist_ok=True)

    # Create test_single_agent directory, clearing it if it exists
    test_dir = results_dir / "test_single_agent"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    # Write the HumanEval sample files
    for filename, content in HUMAN_EVAL_SAMPLES.items():
        with open(test_dir / filename, "w") as f:
            f.write(content)

    return test_dir

def main():
    """Main function to run the test"""
    logger = get_logger()
    logger.configure({"level": "DEBUG",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "console": True,
        "include_caller_info": "rome"})
    logger.info("Starting agent test")

    # Setup test directory with HumanEval samples
    test_dir = setup_test_dir()
    logger.info(f"Test directory created at: {test_dir}")

    # Create a configuration dictionary for the agent
    # Updated to use the new configuration format with class-specific sections
    config = {
        # OpenAIHandler specific configuration
        "OpenAIHandler": {
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 4096,
            "timeout": 30,
        },

        # Agent specific configuration
        "Agent": {
            "repository": str(test_dir.absolute()),
            "fsm_type": "simple"
        },

        # Logger specific configuration
        "Logger": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "console": True
        },

        # SearchAction specific configuration
        "SearchAction": {
            "file_types": [".py"],
            "batch_size": 3,  # Process all 3 files in a single batch
            "selection_criteria": "Select the most interesting code file to improve and complete",
        },
    }

    # Create and initialize the agent
    agent = Agent(
        name="CodeAnalyzer",
        role="You are an expert code analyzer that can identify interesting algorithms and functions",
        config_dict=config
    )
    agent.draw_fsm_graph()

    # Get user input for number of iterations
    results = None
    while True:
        try:
            iterations = int(input("Enter number of iterations to run (0 to quit): "))
            if iterations == 0: raise
        except:
            logger.info("Exiting loop"); break
        results = agent.run_loop(max_iterations=iterations)

    if not results: results = agent.run_loop(max_iterations=0)

    # Shutdown agent
    agent.shutdown()

    # Log and save the results
    logger.info(f"Agent execution completed with {len(results['actions_executed'])} actions")
    logger.info(f"Final state: {results['final_state']}")

    # Save the execution results to a file
    log_dir = Path(logger.get_log_dir())
    try:
        with open(log_dir / "results.json", "w") as f:
            json.dump(results, f, indent=4)

        logger.info(f"Results saved to: {log_dir / 'results.json'}")
    except:
        logger.error(f"Results not serializable: {results}")

if __name__ == "__main__":
    main()
