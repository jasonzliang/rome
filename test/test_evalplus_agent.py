"""
Comprehensive tests for evalplus_single_agent.py
Fixed version with proper class attributes and error handling
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, call, MagicMock
from typing import Dict, List

# Import the module under test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.evalplus_single_agent import EvalPlusBenchmark

class TestBase(unittest.TestCase):
    """Base test class with shared setup and utilities"""

    def setUp(self):
        """Instance-level setup - moved mock_problems here to fix AttributeError"""
        self.mock_problems = {
            "HumanEval/0": {
                "prompt": "def has_close_elements(numbers, threshold):\n    pass",
                "entry_point": "has_close_elements",
                "canonical_solution": "    return any(abs(a-b) < threshold for i,a in enumerate(numbers) for b in numbers[i+1:])",
                "base_input": [{"fn_name": "has_close_elements", "inputs": [[1.0, 2.0], 0.5], "expected": False}],
                "plus_input": [{"fn_name": "has_close_elements", "inputs": [[1.0, 2.8], 0.3], "expected": False}]
            },
            "HumanEval/1": {
                "prompt": "def separate_paren_groups(paren_string):\n    pass",
                "entry_point": "separate_paren_groups",
                "canonical_solution": "    result = []\n    current = []\n    depth = 0\n    for c in paren_string:\n        if c == '(':\n            depth += 1\n        elif c == ')':\n            depth -= 1\n        current.append(c)\n        if depth == 0:\n            result.append(''.join(current))\n            current = []\n    return result",
                "base_input": [{"fn_name": "separate_paren_groups", "inputs": ["()()"], "expected": ["()", "()"]}],
                "plus_input": [{"fn_name": "separate_paren_groups", "inputs": ["((()))"], "expected": ["((()))"]}]
            }
        }

        self.mock_config = {
            "Agent": {"name": "TestAgent", "role": "Test role", "repository": "./"},
            "OpenAIHandler": {"model": "gpt-4o"},
            "Logger": {"level": "INFO"}
        }

        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        self.benchmark_dir = os.path.join(self.temp_dir, "benchmark")

        # Create config file
        with open(self.config_path, 'w') as f:
            f.write("Agent:\n  name: 'TestAgent'\n  role: 'Test role'\n")

    def tearDown(self):
        """Cleanup"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_mock_agent(self, name="TestAgent", log_dir=None, with_solutions=True):
        """Factory method for creating mock agents"""
        log_dir = log_dir or os.path.join(self.temp_dir, "logs")

        class MockAgent:
            def __init__(self):
                self.name = name
                self._log_dir = log_dir
                self.shutdown_called = False
                self.benchmark = None

            def get_log_dir(self):
                os.makedirs(self._log_dir, exist_ok=True)
                return self._log_dir

            def get_id(self): return f"agent_{self.name}_12345"

            def export_config(self, filepath=None):
                if filepath is None:
                    filepath = os.path.join(self.get_log_dir(), f"{self.get_id()}.yaml")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'w') as f:
                    f.write("# Mock config\ntest: true\n")
                return filepath

            def draw_fsm_graph(self): pass
            def shutdown(self): self.shutdown_called = True

            def run_loop(self, max_iterations=10, stop_on_error=False):
                if with_solutions and hasattr(self, 'benchmark') and self.benchmark:
                    self._create_solution_files()
                return {
                    'actions_executed': [{'action': 'test', 'result': True}] * 3,
                    'final_state': 'completed',
                    'errors': []
                }

            def _create_solution_files(self):
                """Create mock solution files during run_loop"""
                if not self.benchmark or not hasattr(self.benchmark, 'problems'):
                    return

                for task_id, problem_info in self.benchmark.problems.items():
                    problem_dir = problem_info["problem_dir"]
                    safe_id = problem_info["safe_id"]
                    solution_file = problem_dir / f"{safe_id}.py"

                    # Get the mock_problems from the test instance through benchmark
                    mock_problems = getattr(self.benchmark, '_test_mock_problems', {})
                    if task_id in mock_problems:
                        entry_point = mock_problems[task_id]['entry_point']
                    else:
                        # Fallback entry point
                        entry_point = "test_function"

                    solution_file.write_text(f"def {entry_point}(*args): return True")

        return MockAgent()

    def setup_benchmark_with_mocks(self, dataset="humaneval", **kwargs):
        """Factory method for creating benchmark with standard mocks"""
        with patch('benchmark.evalplus_single_agent.load_config') as mock_load, \
             patch('benchmark.evalplus_single_agent.merge_with_default_config') as mock_merge, \
             patch('benchmark.evalplus_single_agent.get_human_eval_plus') as mock_dataset:

            mock_load.return_value = self.mock_config
            mock_merge.return_value = {**self.mock_config, "Agent": {**self.mock_config["Agent"], "repository": self.benchmark_dir}}
            mock_dataset.return_value = self.mock_problems

            benchmark = EvalPlusBenchmark(self.config_path, self.benchmark_dir, dataset)
            # Store mock_problems in benchmark for agent access
            benchmark._test_mock_problems = self.mock_problems
            benchmark.setup_problems(**kwargs)
            return benchmark

    def create_eval_subprocess_mocks(self, success=True, scores=None, timeout_on_sanitize=False, timeout_on_eval=False):
        """Factory method for EvalPlus subprocess mocks"""
        if scores is None:
            scores = {"base": {"pass@1": 0.75}, "base_plus_extra": {"pass@1": 0.65}}

        mocks = []

        # Sanitization mock
        if timeout_on_sanitize:
            mocks.append(subprocess.TimeoutExpired(['test'], 300))
        elif success:
            mocks.append(Mock(returncode=0, stdout="Sanitization completed", stderr=""))
        else:
            mocks.append(Mock(returncode=1, stdout="", stderr="Failed"))

        # Evaluation mock (only if sanitization succeeds)
        if not timeout_on_sanitize:
            if timeout_on_eval:
                mocks.append(subprocess.TimeoutExpired(['test'], 1800))
            elif success:
                mocks.append(Mock(returncode=0, stdout=f"Base\n{scores['base']}\nBase + Extra\n{scores['base_plus_extra']}", stderr=""))
            else:
                mocks.append(Mock(returncode=1, stdout="", stderr="Failed"))

        return mocks


class TestEvalPlusBenchmark(TestBase):
    """Core functionality tests"""

    def test_initialization_and_config(self):
        """Test initialization, config loading, and basic properties"""
        benchmark = self.setup_benchmark_with_mocks()

        self.assertEqual(benchmark.dataset, "humaneval")
        self.assertEqual(str(benchmark.benchmark_dir), self.benchmark_dir)
        self.assertIsNotNone(benchmark.config)
        self.assertEqual(len(benchmark.problems), 2)

    def test_task_id_sanitization(self):
        """Test filesystem-safe task ID generation"""
        benchmark = self.setup_benchmark_with_mocks()

        test_cases = [
            ("HumanEval/0", "HumanEval_0"),
            ("HumanEval\\1", "HumanEval_1"),
            ("0_test", "task_0_test")
        ]

        for input_id, expected in test_cases:
            with self.subTest(input_id=input_id):
                self.assertEqual(benchmark._make_safe_task_id(input_id), expected)

    def test_problem_setup_and_filtering(self):
        """Test problem directory creation and filtering options"""
        # Test with num_samples filter
        benchmark = self.setup_benchmark_with_mocks(num_samples=1)
        self.assertEqual(len(benchmark.problems), 1)

        # Test with task_ids filter
        benchmark = self.setup_benchmark_with_mocks(task_ids=["HumanEval/0"])
        self.assertEqual(len(benchmark.problems), 1)
        self.assertIn("HumanEval/0", benchmark.problems)

        # Verify file structure
        problem_dir = Path(self.benchmark_dir) / "HumanEval_0"
        self.assertTrue(problem_dir.exists())
        self.assertTrue((problem_dir / "HumanEval_0.py").exists())
        self.assertTrue((problem_dir / "metadata.json").exists())

    @patch('benchmark.evalplus_single_agent.Agent')
    def test_agent_execution_and_config_export(self, mock_agent_class):
        """Test agent creation, execution, and config export"""
        mock_agent = self.create_mock_agent()
        mock_agent_class.return_value = mock_agent

        benchmark = self.setup_benchmark_with_mocks()
        results = benchmark.run_agent(max_iterations=5)

        # Verify agent interaction
        mock_agent_class.assert_called_once()
        self.assertIn('actions_executed', results)
        self.assertEqual(results['final_state'], 'completed')

    @patch('benchmark.evalplus_single_agent.Agent')
    def test_solution_extraction(self, mock_agent_class):
        """Test solution file extraction"""
        mock_agent = self.create_mock_agent()
        mock_agent_class.return_value = mock_agent

        benchmark = self.setup_benchmark_with_mocks()
        mock_agent.benchmark = benchmark
        benchmark.run_agent()

        solutions = benchmark.extract_solutions()
        self.assertEqual(len(solutions), 2)
        for solution in solutions:
            self.assertIn('task_id', solution)
            self.assertIn('solution', solution)
            self.assertIn('def ', solution['solution'])


class TestEvalPlusIntegration(TestBase):
    """Comprehensive EvalPlus evaluation tests"""

    @patch('benchmark.evalplus_single_agent.subprocess.run')
    @patch('benchmark.evalplus_single_agent.write_jsonl')
    @patch('benchmark.evalplus_single_agent.Agent')
    def test_successful_evaluation_pipeline(self, mock_agent_class, mock_write_jsonl, mock_subprocess):
        """Test complete successful EvalPlus evaluation"""
        mock_agent = self.create_mock_agent()
        mock_agent_class.return_value = mock_agent
        mock_subprocess.side_effect = self.create_eval_subprocess_mocks(success=True)

        benchmark = self.setup_benchmark_with_mocks()
        mock_agent.benchmark = benchmark
        benchmark.run_agent()

        # Create evaluation directory and sanitized file
        eval_dir = Path(mock_agent.get_log_dir()) / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "solutions-sanitized.jsonl").touch()

        mock_solutions = [{"task_id": "HumanEval/0", "solution": "def test(): return True"}]
        results = benchmark.run_evaluation(mock_solutions)

        # Verify subprocess calls
        self.assertEqual(mock_subprocess.call_count, 2)
        sanitize_call, eval_call = mock_subprocess.call_args_list
        self.assertIn("evalplus.sanitize", sanitize_call[0][0])
        self.assertIn("evalplus.evaluate", eval_call[0][0])

        # Verify score parsing
        self.assertIn('scores', results)
        self.assertEqual(results['scores']['base']['pass@1'], 0.75)
        self.assertEqual(results['scores']['base_plus_extra']['pass@1'], 0.65)

    @patch('benchmark.evalplus_single_agent.subprocess.run')
    @patch('benchmark.evalplus_single_agent.Agent')
    def test_evaluation_error_scenarios(self, mock_agent_class, mock_subprocess):
        """Test various evaluation failure scenarios"""
        mock_agent = self.create_mock_agent()
        mock_agent_class.return_value = mock_agent

        benchmark = self.setup_benchmark_with_mocks()
        benchmark.run_agent()

        eval_dir = Path(mock_agent.get_log_dir()) / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        test_cases = [
            # (mock_return, expected_error_substring)
            (Mock(returncode=1, stdout="", stderr="Sanitization failed"), "sanitization failed"),
            (subprocess.TimeoutExpired(['evalplus.sanitize'], 300), "sanitization timed out"),
        ]

        for mock_return, expected_error in test_cases:
            with self.subTest(error=expected_error):
                mock_subprocess.side_effect = [mock_return]
                results = benchmark.run_evaluation([{"task_id": "test", "solution": "pass"}])
                self.assertIn('error', results)
                # More flexible error checking
                error_message = str(results.get('error', ''))
                self.assertTrue(
                    any(keyword in error_message.lower() for keyword in expected_error.lower().split()),
                    f"Expected '{expected_error}' keywords in '{error_message}'"
                )

        # Test exception during subprocess call
        with self.subTest(error="exception"):
            mock_subprocess.side_effect = Exception("Unexpected error")
            results = benchmark.run_evaluation([{"task_id": "test", "solution": "pass"}])
            self.assertIn('error', results)
            self.assertIn('Unexpected error', str(results['error']))

    def test_score_parsing_variations(self):
        """Test evaluation score parsing with different formats"""
        benchmark = self.setup_benchmark_with_mocks()

        test_cases = [
            # (stdout, expected_scores)
            ("Base\n{'pass@1': 0.8}\nBase + Extra\n{'pass@1': 0.7}",
             {"base": {"pass@1": 0.8}, "base_plus_extra": {"pass@1": 0.7}}),
            ("", None),  # Empty output
            ("Invalid format", None)  # Invalid format
        ]

        for stdout, expected in test_cases:
            with self.subTest(stdout=stdout[:20]):
                result = benchmark._parse_evaluation_scores(stdout)
                self.assertEqual(result, expected)


class TestEdgeCasesAndErrors(TestBase):
    """Edge cases and error condition tests"""

    def test_initialization_errors(self):
        """Test initialization with invalid inputs"""
        # Missing config file
        with self.assertRaises(FileNotFoundError):
            EvalPlusBenchmark("/nonexistent/config.yaml", self.benchmark_dir)

        # Invalid dataset - test the error properly
        with patch('benchmark.evalplus_single_agent.load_config') as mock_load, \
             patch('benchmark.evalplus_single_agent.merge_with_default_config') as mock_merge:

            mock_load.return_value = self.mock_config
            mock_merge.return_value = {**self.mock_config, "Agent": {**self.mock_config["Agent"], "repository": self.benchmark_dir}}

            benchmark = EvalPlusBenchmark(self.config_path, self.benchmark_dir, "invalid")
            with self.assertRaises(ValueError) as cm:
                benchmark._get_dataset()
            self.assertIn("Unsupported dataset: invalid", str(cm.exception))

    @patch('benchmark.evalplus_single_agent.Agent')
    def test_missing_solutions_and_save_errors(self, mock_agent_class):
        """Test behavior with missing solutions and save errors"""
        mock_agent = self.create_mock_agent(with_solutions=False)  # No solutions created
        mock_agent_class.return_value = mock_agent

        benchmark = self.setup_benchmark_with_mocks()
        mock_agent.benchmark = benchmark
        benchmark.run_agent()

        # Test missing solution files - manually remove solution files
        for task_id, problem_info in benchmark.problems.items():
            problem_dir = problem_info["problem_dir"]
            safe_id = problem_info["safe_id"]
            solution_file = problem_dir / f"{safe_id}.py"
            if solution_file.exists():
                solution_file.unlink()

        solutions = benchmark.extract_solutions()
        self.assertEqual(len(solutions), 0)

        # Test evaluation with no solutions
        results = benchmark.run_evaluation([])
        self.assertEqual(results['error'], "No solutions found")

        # Test save error by mocking Path.write_text to raise an exception
        with patch.object(Path, 'write_text', side_effect=PermissionError("Permission denied")):
            results_file = benchmark.save_results({}, {})
            self.assertIsNone(results_file)


class TestIntegrationWorkflows(TestBase):
    """End-to-end integration tests"""

    @patch('benchmark.evalplus_single_agent.subprocess.run')
    @patch('benchmark.evalplus_single_agent.write_jsonl')
    @patch('benchmark.evalplus_single_agent.Agent')
    def test_complete_benchmark_pipeline(self, mock_agent_class, mock_write_jsonl, mock_subprocess):
        """Test complete benchmark from setup to results"""
        mock_agent = self.create_mock_agent()
        mock_agent_class.return_value = mock_agent
        mock_subprocess.side_effect = self.create_eval_subprocess_mocks(
            scores={"base": {"pass@1": 0.9}, "base_plus_extra": {"pass@1": 0.85}}
        )

        benchmark = self.setup_benchmark_with_mocks(num_samples=1)  # Limit to 1 problem
        mock_agent.benchmark = benchmark

        # Setup eval directory
        eval_dir = Path(mock_agent.get_log_dir()) / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "solutions-sanitized.jsonl").touch()

        results, results_file = benchmark.run_complete_benchmark(
            max_iterations=10, num_samples=1, run_evaluation=True
        )

        # Comprehensive verification
        self.assertIsNotNone(results)
        self.assertIsNotNone(results_file)
        self.assertTrue(results_file.exists())

        # Check all result components
        self.assertIn('agent_results', results)
        self.assertIn('evaluation_results', results)
        self.assertIn('summary', results)

        summary = results['summary']
        self.assertEqual(summary['problems'], 1)
        self.assertEqual(summary['final_state'], 'completed')
        self.assertIsNotNone(summary['scores'])
        self.assertEqual(summary['scores']['base']['pass@1'], 0.9)

        # Verify agent shutdown
        self.assertTrue(mock_agent.shutdown_called)

    @patch('benchmark.evalplus_single_agent.EvalPlusBenchmark')
    @patch('sys.argv', ['test', 'config.yaml', '/tmp/benchmark', '--dataset', 'mbpp', '--num-samples', '5'])
    def test_main_function_with_args(self, mock_benchmark_class):
        """Test main function argument parsing and execution"""
        mock_benchmark = Mock()
        mock_benchmark_class.return_value = mock_benchmark
        mock_benchmark.run_complete_benchmark.return_value = ({'summary': {}}, Path('/tmp/results'))

        from benchmark.evalplus_single_agent import main

        try:
            main()
        except SystemExit:
            pass

        mock_benchmark_class.assert_called_once_with('config.yaml', '/tmp/benchmark', 'mbpp')
        call_args = mock_benchmark.run_complete_benchmark.call_args
        self.assertEqual(call_args[1]['num_samples'], 5)


class TestEvalPlusSpecificMethods(TestBase):
    """Tests specifically for EvalPlus integration methods"""

    @patch('benchmark.evalplus_single_agent.subprocess.run')
    def test_run_evalplus_command_success(self, mock_subprocess):
        """Test successful EvalPlus command execution"""
        mock_subprocess.return_value = Mock(returncode=0, stdout="Success", stderr="")

        benchmark = self.setup_benchmark_with_mocks()
        eval_dir = Path(self.temp_dir) / "evaluation"
        eval_dir.mkdir(exist_ok=True)

        result = benchmark._run_evalplus_command(
            ["python", "-m", "evalplus.evaluate", "--dataset", "humaneval"],
            eval_dir, "evaluation"
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["stdout"], "Success")
        self.assertEqual(result["return_code"], 0)

    @patch('benchmark.evalplus_single_agent.subprocess.run')
    def test_run_evalplus_command_timeout(self, mock_subprocess):
        """Test EvalPlus command timeout handling"""
        mock_subprocess.side_effect = subprocess.TimeoutExpired(['test'], 300)

        benchmark = self.setup_benchmark_with_mocks()
        eval_dir = Path(self.temp_dir) / "evaluation"
        eval_dir.mkdir(exist_ok=True)

        result = benchmark._run_evalplus_command(
            ["python", "-m", "evalplus.sanitize"], eval_dir, "sanitization", timeout=1
        )

        self.assertIn("error", result)
        self.assertIn("timed out", result["error"])

    @patch('benchmark.evalplus_single_agent.subprocess.run')
    def test_run_evalplus_command_failure(self, mock_subprocess):
        """Test EvalPlus command failure handling"""
        mock_subprocess.return_value = Mock(returncode=1, stdout="", stderr="Command failed")

        benchmark = self.setup_benchmark_with_mocks()
        eval_dir = Path(self.temp_dir) / "evaluation"
        eval_dir.mkdir(exist_ok=True)

        result = benchmark._run_evalplus_command(
            ["python", "-m", "evalplus.sanitize"], eval_dir, "sanitization"
        )

        self.assertIn("error", result)
        self.assertEqual(result["stderr"], "Command failed")

    @patch('benchmark.evalplus_single_agent.subprocess.run')
    @patch('benchmark.evalplus_single_agent.write_jsonl')
    def test_run_evaluation_complete_flow(self, mock_write_jsonl, mock_subprocess):
        """Test complete evaluation flow with both sanitization and evaluation"""
        # Setup mocks for successful sanitization and evaluation
        sanitize_mock = Mock(returncode=0, stdout="Sanitization completed", stderr="")
        evaluate_mock = Mock(returncode=0, stdout="Base\n{'pass@1': 0.8}\nBase + Extra\n{'pass@1': 0.7}", stderr="")
        mock_subprocess.side_effect = [sanitize_mock, evaluate_mock]

        benchmark = self.setup_benchmark_with_mocks()

        # Create mock agent and setup evaluation directory
        mock_agent = self.create_mock_agent()
        benchmark.agent = mock_agent

        eval_dir = Path(mock_agent.get_log_dir()) / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "solutions-sanitized.jsonl").touch()

        solutions = [{"task_id": "HumanEval/0", "solution": "def test(): return True"}]
        results = benchmark.run_evaluation(solutions)

        # Verify both subprocess calls were made
        self.assertEqual(mock_subprocess.call_count, 2)

        # Verify sanitization call
        sanitize_call = mock_subprocess.call_args_list[0]
        self.assertIn("evalplus.sanitize", sanitize_call[0][0])

        # Verify evaluation call
        eval_call = mock_subprocess.call_args_list[1]
        self.assertIn("evalplus.evaluate", eval_call[0][0])
        self.assertIn("--dataset", eval_call[0][0])
        self.assertIn("humaneval", eval_call[0][0])

        # Verify results
        self.assertIn("scores", results)
        self.assertEqual(results["scores"]["base"]["pass@1"], 0.8)
        self.assertEqual(results["scores"]["base_plus_extra"]["pass@1"], 0.7)

    @patch('benchmark.evalplus_single_agent.subprocess.run')
    @patch('benchmark.evalplus_single_agent.write_jsonl')
    def test_run_evaluation_sanitization_failure(self, mock_write_jsonl, mock_subprocess):
        """Test evaluation flow when sanitization fails"""
        # Setup mock for failed sanitization
        sanitize_mock = Mock(returncode=1, stdout="", stderr="Sanitization failed")
        mock_subprocess.side_effect = [sanitize_mock]

        benchmark = self.setup_benchmark_with_mocks()

        # Create mock agent
        mock_agent = self.create_mock_agent()
        benchmark.agent = mock_agent

        solutions = [{"task_id": "HumanEval/0", "solution": "def test(): return True"}]
        results = benchmark.run_evaluation(solutions)

        # Verify only sanitization was called (evaluation should be skipped)
        self.assertEqual(mock_subprocess.call_count, 1)

        # Verify error handling
        self.assertIn("error", results)
        self.assertIn("sanitization failed", results["error"])

    @patch('benchmark.evalplus_single_agent.subprocess.run')
    @patch('benchmark.evalplus_single_agent.write_jsonl')
    def test_run_evaluation_command_verification(self, mock_write_jsonl, mock_subprocess):
        """Test that evalplus.evaluate is called with correct parameters"""
        # Setup mocks for successful sanitization and evaluation
        sanitize_mock = Mock(returncode=0, stdout="Sanitization completed", stderr="")
        evaluate_mock = Mock(returncode=0, stdout="Base\n{'pass@1': 0.85}\nBase + Extra\n{'pass@1': 0.75}", stderr="")
        mock_subprocess.side_effect = [sanitize_mock, evaluate_mock]

        benchmark = self.setup_benchmark_with_mocks(dataset="humaneval")

        # Create mock agent and setup evaluation directory
        mock_agent = self.create_mock_agent()
        benchmark.agent = mock_agent

        eval_dir = Path(mock_agent.get_log_dir()) / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "solutions-sanitized.jsonl").touch()

        solutions = [{"task_id": "HumanEval/0", "solution": "def test(): return True"}]
        results = benchmark.run_evaluation(solutions)

        # Verify both commands were called
        self.assertEqual(mock_subprocess.call_count, 2)

        # Verify sanitization command
        sanitize_call_args = mock_subprocess.call_args_list[0][0][0]
        self.assertIn("python", sanitize_call_args)
        self.assertIn("evalplus.sanitize", sanitize_call_args)
        self.assertIn("--samples", sanitize_call_args)

        # Verify evaluation command with specific parameters
        eval_call_args = mock_subprocess.call_args_list[1][0][0]
        self.assertIn("python", eval_call_args)
        self.assertIn("evalplus.evaluate", eval_call_args)
        self.assertIn("--dataset", eval_call_args)
        self.assertIn("humaneval", eval_call_args)  # Verify correct dataset
        self.assertIn("--samples", eval_call_args)

        # Verify the timeout parameter was passed correctly for evaluation
        eval_call_kwargs = mock_subprocess.call_args_list[1][1]
        self.assertEqual(eval_call_kwargs.get('timeout'), 1800)

        # Verify results contain scores
        self.assertIn("scores", results)
        self.assertEqual(results["scores"]["base"]["pass@1"], 0.85)

    @patch('benchmark.evalplus_single_agent.subprocess.run')
    @patch('benchmark.evalplus_single_agent.write_jsonl')
    def test_run_evaluation_with_mbpp_dataset(self, mock_write_jsonl, mock_subprocess):
        """Test that evalplus.evaluate works with MBPP dataset"""
        # Setup mocks for successful evaluation with MBPP
        sanitize_mock = Mock(returncode=0, stdout="Sanitization completed", stderr="")
        evaluate_mock = Mock(returncode=0, stdout="Base\n{'pass@1': 0.9}\nBase + Extra\n{'pass@1': 0.8}", stderr="")
        mock_subprocess.side_effect = [sanitize_mock, evaluate_mock]

        # Use MBPP dataset
        with patch('benchmark.evalplus_single_agent.get_mbpp_plus') as mock_mbpp:
            mock_mbpp.return_value = self.mock_problems
            benchmark = self.setup_benchmark_with_mocks(dataset="mbpp")

        # Create mock agent
        mock_agent = self.create_mock_agent()
        benchmark.agent = mock_agent

        eval_dir = Path(mock_agent.get_log_dir()) / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "solutions-sanitized.jsonl").touch()

        solutions = [{"task_id": "HumanEval/0", "solution": "def test(): return True"}]
        results = benchmark.run_evaluation(solutions)

        # Verify evaluation command uses MBPP dataset
        eval_call_args = mock_subprocess.call_args_list[1][0][0]
        self.assertIn("--dataset", eval_call_args)
        self.assertIn("mbpp", eval_call_args)  # Verify MBPP dataset

        # Verify results
        self.assertIn("scores", results)
        self.assertEqual(results["scores"]["base"]["pass@1"], 0.9)

    def test_parse_evaluation_scores_edge_cases(self):
        """Test score parsing with various edge cases"""
        benchmark = self.setup_benchmark_with_mocks()

        test_cases = [
            # Valid cases
            ("Base\n{'pass@1': 1.0, 'pass@10': 1.0}\nBase + Extra\n{'pass@1': 0.9, 'pass@10': 0.95}",
             {"base": {"pass@1": 1.0, "pass@10": 1.0}, "base_plus_extra": {"pass@1": 0.9, "pass@10": 0.95}}),

            # Missing Base + Extra
            ("Base\n{'pass@1': 0.5}",
             {"base": {"pass@1": 0.5}}),

            # Invalid eval format
            ("Base\ninvalid_dict_format\nBase + Extra\n{'pass@1': 0.7}",
             {"base_plus_extra": {"pass@1": 0.7}}),

            # Empty output
            ("", None),

            # No Base or Base + Extra markers
            ("Some random output\nwithout markers", None)
        ]

        for stdout, expected in test_cases:
            with self.subTest(stdout=stdout[:50]):
                result = benchmark._parse_evaluation_scores(stdout)
                self.assertEqual(result, expected)


if __name__ == '__main__':
    # Environment setup
    os.environ.setdefault('OPENAI_API_KEY', 'test-key')

    # Run tests with summary
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_class in [TestEvalPlusBenchmark, TestEvalPlusIntegration,
                       TestEdgeCasesAndErrors, TestIntegrationWorkflows,
                       TestEvalPlusSpecificMethods]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)

    # Summary
    print(f"\n{'='*50}")
    print(f"TESTS: {result.testsRun} | FAILURES: {len(result.failures)} | ERRORS: {len(result.errors)}")
    print(f"SUCCESS RATE: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    sys.exit(0 if result.wasSuccessful() else 1)
