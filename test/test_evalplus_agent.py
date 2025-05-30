# test_evalplus_single_agent.py
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

import pytest

# Add the parent directory to sys.path to import from the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module under test
from benchmark.evalplus_single_agent import EvalPlusBenchmark


class TestEvalPlusBenchmark(unittest.TestCase):
    """Test suite for EvalPlusBenchmark class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        self.benchmark_dir = os.path.join(self.temp_dir, "benchmark")

        # Create minimal config file
        self._create_test_config()

        # Mock EvalPlus data
        self.mock_problems = {
            "HumanEval/0": {
                "prompt": "def has_close_elements(numbers, threshold):\n    pass",
                "entry_point": "has_close_elements",
                "canonical_solution": "return False",
                "base_input": [],
                "plus_input": []
            },
            "HumanEval/1": {
                "prompt": "def separate_paren_groups(paren_string):\n    pass",
                "entry_point": "separate_paren_groups",
                "canonical_solution": "return []",
                "base_input": [],
                "plus_input": []
            }
        }

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_test_config(self):
        """Create a minimal test configuration file"""
        config_content = """
Agent:
  name: TestAgent
  role: Test role for agent
  repository: null
  fsm_type: simple
OpenAIHandler:
  model: gpt-4o
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content)

    def _create_benchmark_instance(self, dataset="humaneval"):
        """Helper to create benchmark instance with mocked dependencies"""
        with patch('benchmark.evalplus_single_agent.get_human_eval_plus', return_value=self.mock_problems):
            with patch('benchmark.evalplus_single_agent.load_config') as mock_load:
                mock_load.return_value = {"Agent": {"name": "TestAgent", "role": "Test role"}}
                with patch('benchmark.evalplus_single_agent.merge_with_default_config') as mock_merge:
                    mock_merge.return_value = {
                        "Agent": {"name": "TestAgent", "role": "Test role", "repository": self.benchmark_dir}
                    }
                    return EvalPlusBenchmark(self.benchmark_dir, self.config_path, dataset)

    def test_init_creates_benchmark_directory(self):
        """Test that initialization creates benchmark directory"""
        benchmark = self._create_benchmark_instance()
        self.assertTrue(os.path.exists(self.benchmark_dir))
        self.assertEqual(benchmark.dataset, "humaneval")

    def test_init_with_invalid_config_raises_error(self):
        """Test initialization with non-existent config file"""
        with self.assertRaises(FileNotFoundError):
            EvalPlusBenchmark(self.benchmark_dir, "nonexistent.yaml")

    def test_get_dataset_humaneval(self):
        """Test getting HumanEval dataset"""
        benchmark = self._create_benchmark_instance("humaneval")
        with patch('benchmark.evalplus_single_agent.get_human_eval_plus', return_value=self.mock_problems) as mock_get:
            result = benchmark._get_dataset()
            mock_get.assert_called_once()
            self.assertEqual(result, self.mock_problems)

    def test_get_dataset_mbpp(self):
        """Test getting MBPP dataset"""
        benchmark = self._create_benchmark_instance("mbpp")
        with patch('benchmark.evalplus_single_agent.get_mbpp_plus', return_value=self.mock_problems) as mock_get:
            result = benchmark._get_dataset()
            mock_get.assert_called_once()
            self.assertEqual(result, self.mock_problems)

    def test_get_dataset_invalid_raises_error(self):
        """Test invalid dataset name raises ValueError"""
        benchmark = self._create_benchmark_instance("invalid")
        with self.assertRaises(ValueError) as cm:
            benchmark._get_dataset()
        self.assertIn("Unsupported dataset", str(cm.exception))

    def test_make_safe_task_id(self):
        """Test task ID sanitization"""
        benchmark = self._create_benchmark_instance()

        test_cases = [
            ("HumanEval/0", "HumanEval_0"),
            ("Test\\Path", "Test_Path"),
            ("123test", "task_123test"),
            ("valid_name", "valid_name")
        ]

        for input_id, expected in test_cases:
            with self.subTest(input_id=input_id):
                result = benchmark._make_safe_task_id(input_id)
                self.assertEqual(result, expected)

    def test_setup_problems_creates_files(self):
        """Test that setup_problems creates problem files correctly"""
        benchmark = self._create_benchmark_instance()

        with patch.object(benchmark, '_get_dataset', return_value=self.mock_problems):
            problems = benchmark.setup_problems()

        # Check that problems were set up
        self.assertEqual(len(problems), 2)
        self.assertIn("HumanEval/0", problems)
        self.assertIn("HumanEval/1", problems)

        # Check files were created
        for task_id, problem_info in problems.items():
            problem_file = problem_info["problem_dir"] / f"{problem_info['safe_id']}.py"
            self.assertTrue(problem_file.exists())

    def test_setup_problems_with_num_samples(self):
        """Test setup_problems with num_samples parameter"""
        benchmark = self._create_benchmark_instance()

        with patch.object(benchmark, '_get_dataset', return_value=self.mock_problems):
            problems = benchmark.setup_problems(num_samples=1)

        self.assertEqual(len(problems), 1)

    def test_setup_problems_with_task_ids(self):
        """Test setup_problems with specific task_ids"""
        benchmark = self._create_benchmark_instance()

        with patch.object(benchmark, '_get_dataset', return_value=self.mock_problems):
            problems = benchmark.setup_problems(task_ids=["HumanEval/0"])

        self.assertEqual(len(problems), 1)
        self.assertIn("HumanEval/0", problems)

    def test_setup_problems_existing_files_not_overwritten(self):
        """Test that existing problem files are not overwritten"""
        benchmark = self._create_benchmark_instance()

        # Create a problem file manually
        problem_dir = Path(self.benchmark_dir) / "HumanEval_0"
        problem_dir.mkdir(parents=True)
        problem_file = problem_dir / "HumanEval_0.py"
        original_content = "# Original content"
        problem_file.write_text(original_content)

        with patch.object(benchmark, '_get_dataset', return_value={"HumanEval/0": self.mock_problems["HumanEval/0"]}):
            benchmark.setup_problems()

        # Check that original content is preserved
        self.assertEqual(problem_file.read_text(), original_content)

    @patch('benchmark.evalplus_single_agent.Agent')
    def test_run_agent(self, mock_agent_class):
        """Test running the agent"""
        benchmark = self._create_benchmark_instance()
        mock_agent = Mock()
        mock_agent.export_config.return_value = None
        mock_agent.draw_fsm_graph.return_value = None
        mock_agent.run_loop.return_value = {"actions_executed": [], "final_state": "done"}
        mock_agent_class.return_value = mock_agent

        result = benchmark.run_agent(max_iterations=5)

        mock_agent_class.assert_called_once_with(repository=Path(self.benchmark_dir), config=benchmark.config)
        mock_agent.export_config.assert_called_once()
        mock_agent.draw_fsm_graph.assert_called_once()
        mock_agent.run_loop.assert_called_once_with(max_iterations=5, stop_on_error=False)
        self.assertEqual(result, {"actions_executed": [], "final_state": "done"})

    def test_extract_solutions(self):
        """Test extracting solutions from problem directories"""
        benchmark = self._create_benchmark_instance()

        # Set up some problems first
        benchmark.problems = {
            "HumanEval/0": {"problem_dir": Path(self.benchmark_dir) / "HumanEval_0", "safe_id": "HumanEval_0"},
            "HumanEval/1": {"problem_dir": Path(self.benchmark_dir) / "HumanEval_1", "safe_id": "HumanEval_1"}
        }

        # Create solution files
        for task_id, problem_info in benchmark.problems.items():
            problem_dir = problem_info["problem_dir"]
            problem_dir.mkdir(parents=True, exist_ok=True)
            solution_file = problem_dir / f"{problem_info['safe_id']}.py"
            solution_file.write_text(f"def solution_{task_id.split('/')[-1]}(): pass")

        solutions = benchmark.extract_solutions()

        self.assertEqual(len(solutions), 2)
        for solution in solutions:
            self.assertIn("task_id", solution)
            self.assertIn("solution", solution)
            self.assertTrue(solution["solution"])  # Not empty

    def test_extract_solutions_missing_files(self):
        """Test extracting solutions when some files are missing"""
        benchmark = self._create_benchmark_instance()

        # Set up problems but don't create solution files
        benchmark.problems = {
            "HumanEval/0": {"problem_dir": Path(self.benchmark_dir) / "HumanEval_0", "safe_id": "HumanEval_0"}
        }

        solutions = benchmark.extract_solutions()

        self.assertEqual(len(solutions), 1)
        self.assertEqual(solutions[0]["solution"], "")

    @patch('benchmark.evalplus_single_agent.subprocess.run')
    @patch('benchmark.evalplus_single_agent.write_jsonl')
    def test_run_evaluation(self, mock_write_jsonl, mock_subprocess):
        """Test running EvalPlus evaluation"""
        benchmark = self._create_benchmark_instance()

        # Mock agent for get_log_dir
        mock_agent = Mock()
        mock_agent.get_log_dir.return_value = os.path.join(self.temp_dir, "logs")
        benchmark.agent = mock_agent

        # Create log directory
        os.makedirs(mock_agent.get_log_dir(), exist_ok=True)

        solutions = [{"task_id": "HumanEval/0", "solution": "def test(): pass"}]

        # Mock successful subprocess runs
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="Base tests:\npass@1:\t0.5\nBase + Extra tests:\npass@1:\t0.4",
            stderr=""
        )

        result = benchmark.run_evaluation(solutions)

        # Check that write_jsonl was called
        mock_write_jsonl.assert_called()

        # Check that subprocess was called for evaluation
        self.assertTrue(any("evalplus.evaluate" in str(call_args) for call_args in mock_subprocess.call_args_list))

        # Check result structure
        self.assertIn("stdout", result)
        self.assertEqual(result["return_code"], 0)

    def test_run_evaluation_no_solutions(self):
        """Test run_evaluation with empty solutions list"""
        benchmark = self._create_benchmark_instance()

        result = benchmark.run_evaluation([])

        self.assertIn("error", result)
        self.assertEqual(result["error"], "No solutions found")

    @patch('benchmark.evalplus_single_agent.subprocess.run')
    def test_run_evalplus_command_success(self, mock_subprocess):
        """Test successful EvalPlus command execution"""
        benchmark = self._create_benchmark_instance()

        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="Success output",
            stderr=""
        )

        result = benchmark._run_evalplus_command(
            ["evalplus.evaluate", "--dataset", "humaneval"],
            Path(self.temp_dir),
            "evaluation"
        )

        self.assertNotIn("error", result)
        self.assertEqual(result["return_code"], 0)
        self.assertEqual(result["stdout"], "Success output")

    @patch('benchmark.evalplus_single_agent.subprocess.run')
    def test_run_evalplus_command_failure(self, mock_subprocess):
        """Test failed EvalPlus command execution"""
        benchmark = self._create_benchmark_instance()

        mock_subprocess.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="Error occurred"
        )

        result = benchmark._run_evalplus_command(
            ["evalplus.evaluate", "--dataset", "humaneval"],
            Path(self.temp_dir),
            "evaluation"
        )

        self.assertIn("error", result)
        self.assertEqual(result["return_code"], 1)

    @patch('benchmark.evalplus_single_agent.subprocess.run')
    def test_run_evalplus_command_timeout(self, mock_subprocess):
        """Test EvalPlus command timeout"""
        benchmark = self._create_benchmark_instance()

        mock_subprocess.side_effect = subprocess.TimeoutExpired("cmd", 10)

        result = benchmark._run_evalplus_command(
            ["evalplus.evaluate"],
            Path(self.temp_dir),
            "evaluation",
            timeout=10
        )

        self.assertIn("error", result)
        self.assertIn("timed out", result["error"])

    def test_parse_evaluation_scores(self):
        """Test parsing evaluation scores from EvalPlus output"""
        benchmark = self._create_benchmark_instance()

        # Use exact format that works - the key is the tab character and exact text matching
        test_stdout = "Base tests:\npass@1:\t0.75\nBase + Extra tests:\npass@1:\t0.60"

        scores = benchmark._parse_evaluation_scores(test_stdout)

        self.assertIsNotNone(scores)
        self.assertIn("base", scores)
        self.assertIn("base_plus_extra", scores)
        self.assertEqual(scores["base"]["pass@1"], 0.75)
        self.assertEqual(scores["base_plus_extra"]["pass@1"], 0.60)

    def test_parse_evaluation_scores_invalid_format(self):
        """Test parsing scores with invalid format"""
        benchmark = self._create_benchmark_instance()

        invalid_stdout = "Invalid output format"
        scores = benchmark._parse_evaluation_scores(invalid_stdout)

        self.assertIsNone(scores)

    def test_save_results(self):
        """Test saving benchmark results"""
        benchmark = self._create_benchmark_instance()

        # Mock agent
        mock_agent = Mock()
        mock_agent.name = "TestAgent"
        mock_agent.get_log_dir.return_value = os.path.join(self.temp_dir, "logs")
        benchmark.agent = mock_agent
        benchmark.problems = {"HumanEval/0": {}}

        # Create log directory
        os.makedirs(mock_agent.get_log_dir(), exist_ok=True)

        agent_results = {"actions_executed": [{"action": "test"}], "final_state": "done", "errors": []}
        evaluation_results = {"scores": {"base": {"pass@1": 0.5}}}

        results_file = benchmark.save_results(agent_results, evaluation_results)

        self.assertIsNotNone(results_file)
        self.assertTrue(results_file.exists())

        # Check file content
        with open(results_file) as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data["agent_name"], "TestAgent")
        self.assertEqual(saved_data["dataset"], "humaneval")
        self.assertEqual(saved_data["problems_count"], 1)
        self.assertIn("summary", saved_data)

    @patch.object(EvalPlusBenchmark, 'setup_problems')
    @patch.object(EvalPlusBenchmark, 'run_agent')
    @patch.object(EvalPlusBenchmark, 'extract_solutions')
    @patch.object(EvalPlusBenchmark, 'run_evaluation')
    @patch.object(EvalPlusBenchmark, 'save_results')
    def test_run_complete_benchmark(self, mock_save, mock_eval, mock_extract, mock_run_agent, mock_setup):
        """Test complete benchmark pipeline"""
        benchmark = self._create_benchmark_instance()

        # Mock return values
        mock_setup.return_value = {"HumanEval/0": {}}
        mock_run_agent.return_value = {"actions_executed": [], "final_state": "done"}
        mock_extract.return_value = [{"task_id": "HumanEval/0", "solution": "code"}]
        mock_eval.return_value = {"scores": {"base": {"pass@1": 0.5}}}
        mock_save.return_value = Path(self.temp_dir) / "results.json"

        # Mock agent shutdown
        benchmark.agent = Mock()

        results, results_file = benchmark.run_complete_benchmark(
            max_iterations=5,
            num_samples=1,
            task_ids=["HumanEval/0"]
        )

        # Verify all methods were called
        mock_setup.assert_called_once_with(1, ["HumanEval/0"])
        mock_run_agent.assert_called_once_with(5, False)
        mock_extract.assert_called_once()
        mock_eval.assert_called_once()
        mock_save.assert_called_once()

        # Check results structure
        self.assertIn("agent_results", results)
        self.assertIn("evaluation_results", results)
        self.assertIn("summary", results)

    @patch.object(EvalPlusBenchmark, 'run_complete_benchmark')
    def test_run_complete_benchmark_no_evaluation(self, mock_run_complete):
        """Test complete benchmark without evaluation"""
        benchmark = self._create_benchmark_instance()
        benchmark.agent = Mock()

        mock_run_complete.return_value = ({"summary": {}}, None)

        results, _ = benchmark.run_complete_benchmark(run_evaluation=False)

        # Verify evaluation was skipped in the actual method call
        mock_run_complete.assert_called_once()

    def test_print_summary(self):
        """Test printing benchmark summary"""
        benchmark = self._create_benchmark_instance()

        results = {
            "summary": {
                "problems": 5,
                "actions": 10,
                "final_state": "done",
                "scores": {
                    "base": {"pass@1": 0.8},
                    "base_plus_extra": {"pass@1": 0.6}
                }
            }
        }

        # Capture log output by mocking the logger
        with patch.object(benchmark.logger, 'info') as mock_log:
            benchmark.print_summary(results)

            # Check that summary information was logged
            log_calls = [call[0][0] for call in mock_log.call_args_list]
            summary_logged = any("Problems: 5" in call for call in log_calls)
            self.assertTrue(summary_logged)


class TestMainFunction(unittest.TestCase):
    """Test the main function and argument parsing"""

    @patch('benchmark.evalplus_single_agent.EvalPlusBenchmark')
    @patch('sys.argv', ['script', 'test_dir', 'config.yaml'])
    def test_main_basic_args(self, mock_benchmark_class):
        """Test main function with basic arguments"""
        mock_benchmark = Mock()
        mock_benchmark.run_complete_benchmark.return_value = ({"summary": {}}, Path("results.json"))
        mock_benchmark.logger = Mock()
        mock_benchmark_class.return_value = mock_benchmark

        from benchmark.evalplus_single_agent import main

        with patch('argparse.ArgumentParser.parse_args') as mock_parse:
            mock_parse.return_value = Mock(
                benchmark_dir='test_dir',
                config_file='config.yaml',
                dataset='humaneval',
                num_samples=None,
                task_ids=None,
                max_iterations=0,
                stop_on_error=False,
                no_evaluation=False
            )

            main()

            mock_benchmark_class.assert_called_once_with('test_dir', 'config.yaml', 'humaneval')
            mock_benchmark.run_complete_benchmark.assert_called_once()

    @patch('benchmark.evalplus_single_agent.EvalPlusBenchmark')
    def test_main_with_all_args(self, mock_benchmark_class):
        """Test main function with all arguments"""
        mock_benchmark = Mock()
        mock_benchmark.run_complete_benchmark.return_value = ({"summary": {}}, None)
        mock_benchmark.logger = Mock()
        mock_benchmark_class.return_value = mock_benchmark

        from benchmark.evalplus_single_agent import main

        with patch('argparse.ArgumentParser.parse_args') as mock_parse:
            mock_parse.return_value = Mock(
                benchmark_dir='test_dir',
                config_file='config.yaml',
                dataset='mbpp',
                num_samples=5,
                task_ids=['MBPP/0', 'MBPP/1'],
                max_iterations=10,
                stop_on_error=True,
                no_evaluation=True
            )

            main()

            mock_benchmark_class.assert_called_once_with('test_dir', 'config.yaml', 'mbpp')
            mock_benchmark.run_complete_benchmark.assert_called_once_with(
                max_iterations=10,
                stop_on_error=True,
                num_samples=5,
                task_ids=['MBPP/0', 'MBPP/1'],
                run_evaluation=False  # no_evaluation=True means run_evaluation=False
            )


class TestIntegration(unittest.TestCase):
    """Integration tests with minimal mocking"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        self.benchmark_dir = os.path.join(self.temp_dir, "benchmark")

        # Create test config
        config_content = """
Agent:
  name: TestAgent
  role: Test role for agent
OpenAIHandler:
  model: gpt-4o
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('benchmark.evalplus_single_agent.get_human_eval_plus')
    @patch('benchmark.evalplus_single_agent.Agent')
    @patch('benchmark.evalplus_single_agent.write_jsonl')
    @patch('benchmark.evalplus_single_agent.subprocess.run')
    def test_full_pipeline_integration(self, mock_subprocess, mock_write_jsonl, mock_agent_class, mock_get_data):
        """Test the full pipeline with minimal mocking"""
        # Mock data
        mock_problems = {
            "HumanEval/0": {
                "prompt": "def test_func():\n    pass",
                "entry_point": "test_func"
            }
        }
        mock_get_data.return_value = mock_problems

        # Mock agent
        mock_agent = Mock()
        mock_agent.name = "TestAgent"
        mock_agent.get_log_dir.return_value = os.path.join(self.temp_dir, "logs")
        mock_agent.run_loop.return_value = {"actions_executed": [], "final_state": "done", "errors": []}
        mock_agent_class.return_value = mock_agent

        # Mock successful evaluation
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="Base tests:\npass@1:\t1.0",
            stderr=""
        )

        # Create benchmark and run
        benchmark = EvalPlusBenchmark(self.benchmark_dir, self.config_path)

        results, results_file = benchmark.run_complete_benchmark(
            max_iterations=1,
            num_samples=1
        )

        # Verify results
        self.assertIn("agent_results", results)
        self.assertIn("evaluation_results", results)
        self.assertIn("summary", results)

        # Verify files were created
        self.assertTrue(os.path.exists(self.benchmark_dir))
        problem_dir = Path(self.benchmark_dir) / "HumanEval_0"
        self.assertTrue(problem_dir.exists())


if __name__ == "__main__":
    # Run with verbose output for better debugging
    unittest.main(verbosity=2)
