# benchmark_humaneval.py - Modified EvalPlusBenchmark class
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add the parent directory to sys.path to import from the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rome.agent import Agent
from rome.logger import get_logger
from rome.config import load_config, merge_with_default_config

try:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
except ImportError:
    print("EvalPlus not found. Please install it with: pip install evalplus")
    sys.exit(1)


class EvalPlusBenchmark:
    """Efficient benchmark runner for HumanEval+ with EvalPlus integration"""

    def __init__(self, benchmark_dir: str, config_path: str, dataset: str = "humaneval"):
        self.logger = get_logger()
        self.logger.configure({"level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "console": True})

        self.benchmark_dir = Path(benchmark_dir)
        os.makedirs(self.benchmark_dir, exist_ok=True)
        self.config = self._load_config(config_path)
        self.dataset = dataset.lower()
        self.agent = None
        self.problems = {}

        # Evaluation subprocess attributes
        self.eval_process = None
        self.eval_stop_event = threading.Event()
        self.eval_interval = None

    def _load_config(self, config_path: str) -> Dict:
        """Load and merge configuration"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config = load_config(config_path)
        config = merge_with_default_config(config)
        config.setdefault('Agent', {})['repository'] = str(self.benchmark_dir.absolute())
        return config

    def _get_dataset(self) -> Dict:
        """Get dataset problems"""
        datasets = {'humaneval': get_human_eval_plus, 'mbpp': get_mbpp_plus}
        if self.dataset not in datasets:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        return datasets[self.dataset]()

    def _make_safe_task_id(self, task_id: str) -> str:
        """Convert task_id to filesystem-safe identifier"""
        safe_id = task_id.replace("/", "_").replace("\\", "_")
        return f"task_{safe_id}" if safe_id and safe_id[0].isdigit() else safe_id

    def setup_problems(self, num_samples: Optional[int] = None,
                      task_ids: Optional[List[str]] = None) -> Dict:
        """Setup benchmark directory with problems"""
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)

        problems = self._get_dataset()
        if task_ids:
            problems = {tid: problems[tid] for tid in task_ids if tid in problems}
        if num_samples and num_samples < len(problems):
            problems = dict(list(problems.items())[:num_samples])

        self.problems = {}
        for task_id, problem in problems.items():
            safe_id = self._make_safe_task_id(task_id)
            problem_dir = self.benchmark_dir / safe_id
            problem_dir.mkdir(exist_ok=True)

            problem_file = problem_dir / f"{safe_id}.py"
            if not problem_file.exists():
                problem_file.write_text(problem["prompt"], encoding="utf-8")

            self.problems[task_id] = {"problem_dir": problem_dir, "safe_id": safe_id}

        self.logger.info(f"Setup {len(self.problems)} problems in {self.benchmark_dir}")
        return self.problems

    def run_agent(self, max_iterations: int = 10, stop_on_error: bool = False) -> Dict:
        """Run agent on benchmark problems"""
        self.agent = Agent(repository=self.benchmark_dir, config=self.config)

        try:
            self.agent.export_config()
            self.agent.draw_fsm_graph()
        except Exception as e:
            self.logger.error(f"Agent setup error: {e}")

        self.logger.info(f"Running agent for {max_iterations} iterations")
        results = self.agent.run_loop(max_iterations=max_iterations, stop_on_error=stop_on_error)

        self.logger.info(f"Agent completed: {len(results.get('actions_executed', []))} actions, "
                        f"final state: {results.get('final_state')}")
        return results

    def extract_solutions(self) -> List[Dict]:
        """Extract solutions from problem directories"""
        solutions = []
        for task_id, problem_info in self.problems.items():
            problem_dir = problem_info["problem_dir"]
            safe_id = problem_info["safe_id"]

            solution_file = problem_dir / f"{safe_id}.py"
            if solution_file.exists():
                solutions.append({
                    "task_id": task_id,
                    "solution": solution_file.read_text(encoding="utf-8")
                })
            else:
                solutions.append({"task_id": task_id, "solution": ""})

        return solutions

    def run_evaluation(self, solutions: List[Dict]) -> Dict:
        """Run EvalPlus evaluation on solutions"""
        if not solutions:
            return {"error": "No solutions found"}

        eval_dir = Path(self.agent.get_log_dir()) / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Save solutions
        solutions_file = eval_dir / "solutions.jsonl"
        write_jsonl(str(solutions_file), solutions)

        # Remove existing eval results
        for eval_results_file in eval_dir.glob("*.eval_results.json"):
            eval_results_file.unlink()

        # Try sanitization first
        try:
            subprocess.run(["evalplus.sanitize", "--samples", str(solutions_file)],
                          cwd=eval_dir, capture_output=True, timeout=300)
            sanitized_file = eval_dir / "solutions-sanitized.jsonl"
            if sanitized_file.exists():
                solutions_file = sanitized_file
        except:
            pass

        # Run evaluation
        try:
            result = subprocess.run(
                ["evalplus.evaluate", "--dataset", self.dataset, "--samples", str(solutions_file)],
                cwd=eval_dir, capture_output=True, text=True, timeout=1800
            )

            if result.returncode == 0:
                scores = self._parse_evaluation_scores(result.stdout)
                return {"stdout": result.stdout, "scores": scores} if scores else {"stdout": result.stdout}
            else:
                return {"error": "evaluation failed", "stderr": result.stderr}

        except subprocess.TimeoutExpired:
            return {"error": "evaluation timeout"}
        except Exception as e:
            return {"error": str(e)}

    def _parse_evaluation_scores(self, stdout: str) -> Optional[Dict]:
        """Parse pass@k scores from EvalPlus output"""
        try:
            lines = stdout.strip().split('\n')
            scores = {}

            for i, line in enumerate(lines[:-1]):
                lower_line = line.lower()
                next_line = lines[i + 1].strip()

                if next_line.startswith("pass@") and '\t' in next_line:
                    key, value = next_line.split('\t', 1)
                    score = {key.rstrip(':'): float(value)}

                    if "base tests" in lower_line:
                        scores["base"] = score
                    elif "base + extra" in lower_line:
                        scores["base_plus_extra"] = score

            return scores or None
        except:
            return None

    def _periodic_evaluation_worker(self):
        """Worker thread for periodic evaluation using subprocess"""
        while not self.eval_stop_event.wait(self.eval_interval):
            try:
                solutions = self.extract_solutions()
                if solutions:
                    evaluation_results = self.run_evaluation(solutions)
                    if "scores" in evaluation_results:
                        scores = evaluation_results["scores"]
                        score_summary = []
                        for score_type, metrics in scores.items():
                            if "pass@1" in metrics:
                                score_summary.append(f"{score_type}: {metrics['pass@1']:.3f}")
                        if score_summary:
                            self.logger.info(f"Periodic eval - {', '.join(score_summary)}")
            except Exception as e:
                self.logger.error(f"Periodic evaluation failed: {e}")

    def start_periodic_evaluation(self, eval_interval: int):
        """Start periodic evaluation in subprocess"""
        if not eval_interval or eval_interval <= 0:
            return

        self.eval_interval = eval_interval
        self.eval_stop_event.clear()

        eval_thread = threading.Thread(target=self._periodic_evaluation_worker, daemon=True)
        eval_thread.start()
        self.logger.info(f"Started periodic evaluation (interval: {eval_interval}s)")

    def stop_periodic_evaluation(self):
        """Stop periodic evaluation"""
        if self.eval_interval:
            self.eval_stop_event.set()
            self.logger.info("Stopped periodic evaluation")

    def save_results(self, agent_results: Dict, evaluation_results: Dict) -> Path:
        """Save complete benchmark results"""
        eval_dir = Path(self.agent.get_log_dir()) / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "agent_name": self.agent.name,
            "dataset": self.dataset,
            "benchmark_directory": str(self.benchmark_dir),
            "problems_count": len(self.problems),
            "agent_results": agent_results,
            "evaluation_results": evaluation_results,
            "summary": {
                "total_actions": len(agent_results.get('actions_executed', [])),
                "final_state": agent_results.get('final_state', 'unknown'),
                "errors": len(agent_results.get('errors', [])),
                "evaluation_success": "scores" in evaluation_results,
                "scores": evaluation_results.get("scores")
            }
        }

        results_file = eval_dir / "benchmark_results.json"
        try:
            results_file.write_text(json.dumps(results, indent=4, default=str))
            return results_file
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return None

    def run_complete_benchmark(self, max_iterations: int = 10, stop_on_error: bool = False,
                              num_samples: Optional[int] = None, task_ids: Optional[List[str]] = None,
                              run_evaluation: bool = True, eval_interval: Optional[int] = None) -> Tuple[Dict, Path]:
        """Run complete benchmark pipeline with optional periodic evaluation"""
        try:
            # Setup problems
            self.setup_problems(num_samples, task_ids)

            # Start periodic evaluation if requested
            if eval_interval:
                self.start_periodic_evaluation(eval_interval)

            # Run agent
            agent_results = self.run_agent(max_iterations, stop_on_error)

            # Stop periodic evaluation
            if eval_interval:
                self.stop_periodic_evaluation()

            # Run final evaluation
            evaluation_results = {}
            if run_evaluation:
                solutions = self.extract_solutions()
                evaluation_results = self.run_evaluation(solutions)

            # Save and return results
            results_file = self.save_results(agent_results, evaluation_results)
            if self.agent:
                self.agent.shutdown()

            return {
                "agent_results": agent_results,
                "evaluation_results": evaluation_results,
                "summary": {
                    "problems": len(self.problems),
                    "actions": len(agent_results.get('actions_executed', [])),
                    "final_state": agent_results.get('final_state'),
                    "scores": evaluation_results.get("scores")
                }
            }, results_file

        except Exception as e:
            if eval_interval:
                self.stop_periodic_evaluation()
            if self.agent:
                self.agent.shutdown()
            raise

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
            self.logger.info("\nEvalPlus evaluation results:")
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
    parser.add_argument("--num-samples", type=int, help="Number of samples to include")
    parser.add_argument("--task-ids", nargs="+", help="Specific task IDs to include")
    parser.add_argument("--max-iterations", type=int, default=10, help="Iterations for agent to run")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop agent if exception thrown")
    parser.add_argument("--no-evaluation", action="store_true", help="Skip final evaluation")
    parser.add_argument("--eval-interval", type=int, help="Periodic evaluation interval in seconds")

    args = parser.parse_args()

    benchmark = EvalPlusBenchmark(args.benchmark_dir, args.config_file, args.dataset)

    results, results_file = benchmark.run_complete_benchmark(
        max_iterations=args.max_iterations,
        stop_on_error=args.stop_on_error,
        num_samples=args.num_samples,
        task_ids=args.task_ids,
        run_evaluation=not args.no_evaluation,
        eval_interval=args.eval_interval
    )

    benchmark.print_summary(results)
    if results_file:
        benchmark.logger.info(f"\nDetailed results: {results_file}")


if __name__ == "__main__":
    main()
