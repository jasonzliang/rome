# evalplus_single_agent_refactored.py - Refactored with EvalplusEvaluator
import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add the parent directory to sys.path to import from the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rome.agent import Agent
from rome.logger import get_logger
from rome.config import load_config, merge_with_default_config
from benchmark.evalplus_evaluator import EvalplusEvaluator, PeriodicEvaluator

class EvalPlusBenchmark:
    """Simple benchmark runner for HumanEval+ with modular evaluation"""

    def __init__(self, benchmark_dir: str, config_path: str, dataset: str = "humaneval"):
        self.logger = get_logger()
        self.logger.configure({
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "console": True
        })

        self.benchmark_dir = Path(benchmark_dir)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config(config_path)
        self.dataset = dataset.lower()
        self.agent = None

        # Initialize evaluator with benchmark directory
        # FIXED: Removed logger parameter
        self.evaluator = EvalplusEvaluator(
            benchmark_dir=self.benchmark_dir,
            dataset=self.dataset,
        )
        self.periodic_evaluator = None

    def _load_config(self, config_path: str) -> Dict:
        """Load and merge configuration"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config = load_config(config_path)
        config = merge_with_default_config(config)
        config['Agent']['repository'] = str(self.benchmark_dir.absolute())
        return config

    def _setup_periodic_evaluation(self, eval_interval: int):
        """Setup periodic evaluation"""
        if eval_interval <= 0:
            return

        self.periodic_evaluator = PeriodicEvaluator(
            evaluator=self.evaluator,
            interval=eval_interval
        )

    def _run_agent(self, max_iterations: int, stop_on_error: bool) -> Dict:
        """Run agent with periodic evaluation using callback system"""
        agent_config = self.config.get('Agent', {})

        self.agent = Agent(
            name=agent_config['name'],
            role=agent_config['role'],
            repository=str(self.benchmark_dir),
            config=self.config
        )

        # Set up periodic evaluation callback if enabled
        if self.periodic_evaluator:
            self.periodic_evaluator.set_callback(self.agent.set_callback)

        # Run agent normally - callback will handle periodic evaluation
        return self.agent.run_loop(max_iterations=max_iterations, stop_on_error=stop_on_error)

    def _save_results(self, agent_results: Dict, evaluation_results: Dict) -> Optional[Path]:
        """Save benchmark results"""
        results = {
            "agent_name": self.agent.name if self.agent else "unknown",
            "dataset": self.dataset,
            "benchmark_directory": str(self.benchmark_dir),
            "problems_count": len(self.evaluator.problems),
            "agent_results": agent_results,
            "evaluation_results": evaluation_results,
            "summary": {
                "total_actions": agent_results.get('execution_stats', {}).get('actions_executed', 0),
                "final_state": agent_results.get('agent_info', {}).get('current_state', 'unknown'),
                "evaluation_success": evaluation_results.get("success", False),
                "scores": evaluation_results.get("scores")
            }
        }

        results_file = self.evaluator.log_dir / "benchmark.results.json"
        try:
            results_file.write_text(json.dumps(results, indent=4, default=str))
            return results_file
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return None

    def run_benchmark(self, max_iterations: int = 10, stop_on_error: bool = False,
                              num_samples: Optional[int] = None, task_ids: Optional[List[str]] = None,
                              run_evaluation: bool = True, eval_interval: Optional[int] = None) -> Tuple[Dict, Path]:
        """Run complete benchmark pipeline with modular evaluation"""
        try:
            # Setup problems
            problems_dict = self.evaluator.setup_problems(num_samples, task_ids)
            if not problems_dict:
                raise ValueError("No problems were set up")

            # Setup periodic evaluation if requested
            if eval_interval and run_evaluation:
                self._setup_periodic_evaluation(eval_interval)

            # Run agent with evaluation
            agent_results = self._run_agent(max_iterations, stop_on_error)

            # Run final evaluation
            evaluation_results = {}
            if run_evaluation:
                self.logger.info("Running final evaluation...")
                evaluation_results = self.evaluator.evaluate()

            # Save results
            results_file = self._save_results(agent_results, evaluation_results)

            return {
                "agent_results": agent_results,
                "evaluation_results": evaluation_results,
                "summary": {
                    "problems": len(self.evaluator.problems),
                    "actions": agent_results.get('execution_stats', {}).get('actions_executed', 0),
                    "final_state": agent_results.get('agent_info', {}).get('current_state'),
                    "scores": evaluation_results.get("scores")
                }
            }, results_file

        except Exception as e:
            self.logger.error(f"Benchmark execution failed: {e}")
            raise
        finally:
            if self.agent:
                self.agent.shutdown()

    def print_summary(self, results: Dict):
        """Print benchmark summary"""
        summary = results.get("summary", {})
        scores = summary.get("scores", {})

        self.logger.info("="*80)
        self.logger.info("EvalPlus benchmark summary")
        self.logger.info("="*80)
        self.logger.info(f"Dataset: {self.dataset}")
        self.logger.info(f"Problems: {summary.get('problems', 0)}")
        self.logger.info(f"Actions: {summary.get('actions', 0)}")
        self.logger.info(f"Final State: {summary.get('final_state', 'unknown')}")

        if scores:
            self.logger.info("\nEvaluation results:")
            for score_type, metrics in scores.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        self.logger.info(f"{score_type.replace('_', ' ').title()} {metric}: {value}")
        self.logger.info("="*80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Benchmark agent on HumanEval+ dataset")
    parser.add_argument("benchmark_dir", help="Path to benchmark directory")
    parser.add_argument("config_file", help="Path to agent configuration YAML file")

    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--eval-interval", type=int, default=1800, help="Periodic evaluation interval in seconds")
    parser.add_argument("--max-iterations", type=int, default=0, help="Iterations for agent to run")
    parser.add_argument("--no-evaluation", action="store_true", help="Skip evaluation")
    parser.add_argument("--num-samples", type=int, help="Number of samples to include")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop agent if exception thrown")
    parser.add_argument("--task-ids", nargs="+", help="Specific task IDs to include")

    args = parser.parse_args()

    benchmark = EvalPlusBenchmark(args.benchmark_dir, args.config_file, args.dataset)

    results, results_file = benchmark.run_benchmark(
        max_iterations=args.max_iterations,
        stop_on_error=args.stop_on_error,
        num_samples=args.num_samples,
        task_ids=args.task_ids,
        run_evaluation=not args.no_evaluation,
        eval_interval=args.eval_interval
    )

    benchmark.print_summary(results)
    if results_file:
        benchmark.logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
