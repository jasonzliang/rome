# evalplus_multi_agent_refactored.py - Compact multi-agent benchmark
import argparse
import asyncio
import json
import pprint
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rome.config import LOG_DIR_NAME, merge_with_default_config
from rome.logger import get_logger
from rome.multi_agent import MultiAgent
from benchmark.evalplus_evaluator import EvalplusEvaluator, PeriodicEvaluator


class MultiAgentEvalPlusBenchmark:
    """Compact multi-agent benchmark with periodic evaluation"""

    def __init__(self, benchmark_dir: str, config_path: str, agents_config_path: Optional[str] = None,
                 dataset: str = "humaneval", eval_interval: Optional[int] = None):
        self.benchmark_dir = Path(benchmark_dir)
        self.config_path = Path(config_path)
        self.dataset = dataset
        self.eval_interval = eval_interval

        # Initialize components
        self.logger = get_logger()
        self.logger.configure({
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "console": True
        })

        self.config = self._load_config()
        self.agents_config_path = self._resolve_agents_config(agents_config_path)
        self.evaluator = self._create_evaluator()
        self.periodic_evaluator = PeriodicEvaluator(self.evaluator, eval_interval) if eval_interval else None
        self.multi_agent = None

        self.logger.info(f"Multi-agent benchmark initialized: {self.dataset}, {self.agents_config_path}")

    def _load_config(self) -> Dict:
        """Load and validate YAML configuration"""
        self._validate_file(self.config_path, "Config file")

        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f) or {}
            config['_config_file_path'] = str(self.config_path)
            config = merge_with_default_config(config)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {self.config_path}: {e}")

    def _resolve_agents_config(self, agents_config_path: Optional[str]) -> str:
        """Resolve agents config with fallback to config file"""
        # Command line argument takes priority
        if agents_config_path:
            self._validate_file(agents_config_path, "Agents config")
            return Path(agents_config_path)

        # Fallback to config file
        config_path = self.config.get('MultiAgent', {}).get('agent_role_json')
        if not config_path:
            raise ValueError("Agents config required: use --agents-config or set MultiAgent.agent_role_json")

        config_path = Path(config_path)
        self._validate_file(config_path, "Agents config from config file")
        return config_path

    def _get_eval_dir(self) -> Path:
        """Get evaluation directory"""
        return self.benchmark_dir / LOG_DIR_NAME / "evaluation"

    def _create_evaluator(self) -> EvalplusEvaluator:
        """Create evaluator with proper directory structure"""
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        return EvalplusEvaluator(self.benchmark_dir, self._get_eval_dir(), self.dataset)

    def _validate_file(self, path: Path | str, description: str):
        """Validate file exists with descriptive error"""
        if not Path(path).exists():
            raise FileNotFoundError(f"{description} not found: {path}")

    def _format_scores(self, scores: Dict, prefix: str = "") -> str:
        """Format scores for logging"""
        return ", ".join(f"{k.replace('_', '+').title()}: {v.get('pass@1', 0):.3f}"
                        for k, v in scores.items() if isinstance(v, dict) and 'pass@1' in v) or "No scores"

    def _format_agent_result(self, name: str, result: Dict) -> Dict:
        """Format single agent result"""
        if 'error' in result:
            error = result['error']
            return {
                "success": False, "agent_name": name,
                "error": error.get('error_message', str(error)),
                "error_type": error.get('error_type', 'unknown')
            }
        return {"success": True, "agent_name": name, "agent_results": result}

    def run_agents(self, max_iterations: int, stop_on_error: bool) -> List[Dict]:
        """Run agents with error handling and result formatting"""
        try:
            self.multi_agent = MultiAgent(self.agents_config_path, str(self.benchmark_dir), self.config)
            results = self.multi_agent.run_loop(max_iterations=max_iterations, timeout_per_agent=0)

            # Format results using helper
            agent_results = [self._format_agent_result(name, result)
                           for name, result in results.get('agent_results', {}).items()]

            success_count = sum(1 for r in agent_results if r["success"])
            self.logger.info(f"Agents completed: {success_count}/{len(agent_results)}")

            if not agent_results:
                raise ValueError("No agents executed")
            return agent_results

        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}")
            raise
        finally:
            if self.multi_agent:
                try:
                    self.multi_agent.shutdown()
                except Exception as e:
                    self.logger.warning(f"Shutdown error: {e}")

    def save_results(self, agent_results: List[Dict], evaluation_results: Dict) -> Path:
        """Save results with backup fallback"""
        results = {
            "benchmark_type": "multi_agent", "dataset": self.dataset,
            "benchmark_dir": str(self.benchmark_dir), "agents_config": self.agents_config_path,
            "problems_count": len(self.evaluator.problems), "total_agents": len(agent_results),
            "successful_agents": sum(1 for r in agent_results if r["success"]),
            "evaluation_results": evaluation_results, "agent_results": agent_results,
            "scores": evaluation_results.get("scores")
        }

        self.logger.info(f"Benchmark results:\n{pprint.pformat(results)}")

        # Try primary location, fallback to backup
        path = self.benchmark_dir / LOG_DIR_NAME / "multi_agent_results.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Results saved: {path}")
            return path
        except Exception as e:
            self.logger.warning(f"Save failed for {path}: {e}")

        raise RuntimeError("Failed to save results to any location")


    async def run_benchmark_async(self, max_iterations: int = 10, stop_on_error: bool = False,
                                 num_samples: Optional[int] = None, task_ids: Optional[List[str]] = None,
                                 run_evaluation: bool = True) -> Tuple[Dict, Path]:
        """Complete async benchmark pipeline with cleanup"""
        agent_results = []
        evaluation_results = {}

        try:
            # Setup and run
            problems_dict = self.evaluator.setup_problems(num_samples, task_ids)
            if not problems_dict:
                raise ValueError("No problems were set up")

            # Start periodic evaluation
            if run_evaluation and self.periodic_evaluator:
                await self.periodic_evaluator.start_async()

            # Run agents and evaluate
            agent_results = self.run_agents(max_iterations, stop_on_error)

            if run_evaluation:
                self.evaluator.evaluate(blocking=False)

            # Save and return results
            results_file = self.save_results(agent_results, evaluation_results)

            return agent_results, results_file

        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            # Try to save partial results
            if agent_results:
                try:
                    self.save_results(agent_results, evaluation_results)
                except Exception:
                    pass
            raise
        finally:
            # Cleanup
            if self.periodic_evaluator:
                await self.periodic_evaluator.stop_async()


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(description="Multi-agent EvalPlus benchmark")
    parser.add_argument("benchmark_dir", help="Shared benchmark directory")
    parser.add_argument("config_file", help="Agent configuration YAML")
    parser.add_argument("--agents-config", help="Agents configuration JSON")
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--num-samples", type=int, help="Number of samples")
    parser.add_argument("--task-ids", nargs="+", help="Specific task IDs")
    parser.add_argument("--eval-interval", type=int, help="Evaluation interval (seconds)")
    parser.add_argument("--max-iterations", type=int, default=0, help="Max iterations per agent")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on errors")
    parser.add_argument("--no-evaluation", action="store_true", help="Skip evaluation")
    return parser


def main():
    """Main entry point with error handling"""
    parser = create_parser()
    args = parser.parse_args()

    benchmark = MultiAgentEvalPlusBenchmark(
        args.benchmark_dir, args.config_file, args.agents_config,
        args.dataset, args.eval_interval
    )

    results_file = asyncio.run(benchmark.run_benchmark_async(
        max_iterations=args.max_iterations, stop_on_error=args.stop_on_error,
        num_samples=args.num_samples, task_ids=args.task_ids,
        run_evaluation=not args.no_evaluation
    ))
    # benchmark.logger.info(f"Results: {results_file}")

if __name__ == "__main__":
    main()
