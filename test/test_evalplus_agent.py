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


class TestEvalPlusBenchmark(unittest.TestCase):
    """Test suite for EvalPlusBenchmark class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        self.benchmark_dir = os.path.join(self.temp_dir, "benchmark")

        # Create minimal config file
        self._create_test_config()

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
LLMHandler:
  model: gpt-4o
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content)

    def _create_benchmark_instance(self, dataset="humaneval"):
        """Helper to create benchmark instance with mocked evaluator"""
        with patch('benchmark.evalplus_evaluator.get_human_eval_plus'), \
             patch('benchmark.evalplus_evaluator.get_mbpp_plus'):
            from benchmark.evalplus_single_agent import EvalPlusBenchmark
            return EvalPlusBenchmark(self.benchmark_dir, self.config_path, dataset)

    def test_init_creates_benchmark_directory(self):
        """Test that initialization creates benchmark directory"""
        benchmark = self._create_benchmark_instance()
        self.assertTrue(os.path.exists(self.benchmark_dir))
        self.assertEqual(benchmark.dataset, "humaneval")

    def test_init_with_invalid_config_raises_error(self):
        """Test initialization with non-existent config file"""
        with patch('benchmark.evalplus_evaluator.get_human_eval_plus'), \
             patch('benchmark.evalplus_evaluator.get_mbpp_plus'):
            from benchmark.evalplus_single_agent import EvalPlusBenchmark
            with self.assertRaises(FileNotFoundError):
                EvalPlusBenchmark(self.benchmark_dir, "nonexistent.yaml")

    def test_load_config(self):
        """Test that config is loaded and merged properly"""
        benchmark = self._create_benchmark_instance()
        self.assertIn("Agent", benchmark.config)
        self.assertEqual(benchmark.config["Agent"]["name"], "TestAgent")
        # repository should be set to benchmark_dir
        self.assertEqual(benchmark.config["Agent"]["repository"], str(Path(self.benchmark_dir).absolute()))

    def test_evaluator_setup_problems(self):
        """Test that evaluator.setup_problems is called via benchmark"""
        benchmark = self._create_benchmark_instance()
        mock_problems = {"HumanEval/0": {"dir": Path("/tmp/test"), "safe_id": "HumanEval_0"}}

        with patch.object(benchmark.evaluator, 'setup_problems', return_value=mock_problems) as mock_setup:
            # Call setup_problems via the evaluator
            problems = benchmark.evaluator.setup_problems()
            mock_setup.assert_called_once()
            self.assertEqual(len(problems), 1)

    def test_evaluator_safe_task_id(self):
        """Test task ID sanitization via evaluator"""
        benchmark = self._create_benchmark_instance()

        test_cases = [
            ("HumanEval/0", "HumanEval_0"),
            ("Test\\Path", "Test_Path"),
            ("123test", "task_123test"),
            ("valid_name", "valid_name")
        ]

        for input_id, expected in test_cases:
            with self.subTest(input_id=input_id):
                result = benchmark.evaluator._safe_task_id(input_id)
                self.assertEqual(result, expected)

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

    @patch('benchmark.evalplus_single_agent.Agent')
    def test_run_agent(self, mock_agent_class):
        """Test running the agent"""
        benchmark = self._create_benchmark_instance()
        mock_agent = Mock()
        mock_agent.run_loop.return_value = {"actions_executed": [], "final_state": "done"}
        mock_agent_class.return_value = mock_agent

        result = benchmark._run_agent(max_iterations=5, stop_on_error=False)

        mock_agent_class.assert_called_once()
        mock_agent.run_loop.assert_called_once_with(max_iterations=5, stop_on_error=False)
        self.assertEqual(result, {"actions_executed": [], "final_state": "done"})

    def test_save_results(self):
        """Test saving benchmark results"""
        benchmark = self._create_benchmark_instance()

        # Setup mock agent
        mock_agent = Mock()
        mock_agent.name = "TestAgent"
        benchmark.agent = mock_agent
        benchmark.evaluator.problems = {"HumanEval/0": {}}

        agent_results = {"execution_stats": {"actions_executed": 5}, "agent_info": {"current_state": "done"}}
        evaluation_results = {"scores": {"base": {"pass@1": 0.5}}}

        results_file = benchmark._save_results(agent_results, evaluation_results)

        self.assertIsNotNone(results_file)
        self.assertTrue(results_file.exists())

        # Check file content
        with open(results_file) as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data["agent_name"], "TestAgent")
        self.assertEqual(saved_data["dataset"], "humaneval")

    def test_run_benchmark_orchestration(self):
        """Test that run_benchmark orchestrates properly"""
        benchmark = self._create_benchmark_instance()

        mock_problems = {"HumanEval/0": {"dir": Path("/tmp/test"), "safe_id": "HumanEval_0"}}
        mock_eval_results = {"scores": {"base": {"pass@1": 0.5}}, "success": True}
        mock_agent_results = {"execution_stats": {"actions_executed": 5}, "agent_info": {"current_state": "done"}}

        with patch.object(benchmark.evaluator, 'setup_problems', return_value=mock_problems), \
             patch.object(benchmark, '_run_agent', return_value=mock_agent_results), \
             patch.object(benchmark.evaluator, 'evaluate', return_value=mock_eval_results), \
             patch.object(benchmark, '_save_results', return_value=Path("/tmp/results.json")):

            benchmark.agent = Mock()  # Mock for shutdown
            results, results_file = benchmark.run_benchmark(
                max_iterations=5,
                num_problems=1,
                task_ids=["HumanEval/0"]
            )

            self.assertIn("agent_results", results)
            self.assertIn("evaluation_results", results)
            self.assertIn("summary", results)


class TestMainFunction(unittest.TestCase):
    """Test the main function and argument parsing"""

    @patch('benchmark.evalplus_evaluator.get_human_eval_plus')
    @patch('benchmark.evalplus_evaluator.get_mbpp_plus')
    @patch('benchmark.evalplus_single_agent.EvalPlusBenchmark')
    @patch('sys.argv', ['script', 'test_dir', 'config.yaml'])
    def test_main_basic_args(self, mock_benchmark_class, mock_mbpp, mock_humaneval):
        """Test main function with basic arguments"""
        mock_benchmark = Mock()
        mock_benchmark.run_benchmark.return_value = ({"summary": {}}, Path("results.json"))
        mock_benchmark.logger = Mock()
        mock_benchmark_class.return_value = mock_benchmark

        from benchmark.evalplus_single_agent import main

        with patch('argparse.ArgumentParser.parse_args') as mock_parse:
            mock_parse.return_value = Mock(
                benchmark_dir='test_dir',
                config_file='config.yaml',
                dataset='humaneval',
                num_problems=None,
                task_ids=None,
                max_iterations=0,
                no_stop_on_error=False,
                no_evaluation=False,
                eval_interval=600
            )

            main()

            mock_benchmark_class.assert_called_once_with('test_dir', 'config.yaml', 'humaneval')
            mock_benchmark.run_benchmark.assert_called_once()

    @patch('benchmark.evalplus_evaluator.get_human_eval_plus')
    @patch('benchmark.evalplus_evaluator.get_mbpp_plus')
    @patch('benchmark.evalplus_single_agent.EvalPlusBenchmark')
    def test_main_with_all_args(self, mock_benchmark_class, mock_mbpp, mock_humaneval):
        """Test main function with all arguments"""
        mock_benchmark = Mock()
        mock_benchmark.run_benchmark.return_value = ({"summary": {}}, None)
        mock_benchmark.logger = Mock()
        mock_benchmark_class.return_value = mock_benchmark

        from benchmark.evalplus_single_agent import main

        with patch('argparse.ArgumentParser.parse_args') as mock_parse:
            mock_parse.return_value = Mock(
                benchmark_dir='test_dir',
                config_file='config.yaml',
                dataset='mbpp',
                num_problems=5,
                task_ids=['MBPP/0', 'MBPP/1'],
                max_iterations=10,
                no_stop_on_error=True,
                no_evaluation=True,
                eval_interval=600
            )

            main()

            mock_benchmark_class.assert_called_once_with('test_dir', 'config.yaml', 'mbpp')
            mock_benchmark.run_benchmark.assert_called_once()


if __name__ == "__main__":
    # Run with verbose output for better debugging
    unittest.main(verbosity=2)
