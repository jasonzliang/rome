# evalplus_single_agent.py - Clean and simple EvalPlus benchmark
import argparse
import json
import os
import re
import subprocess
import sys
import traceback
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
    """Simple benchmark runner for HumanEval+ with optional background evaluation"""

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
        self.problems = {}

        # Simple periodic evaluation state
        self.eval_interval = None
        self.eval_process = None
        self.last_eval_time = 0

    def _load_config(self, config_path: str) -> Dict:
        """Load and merge configuration"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config = load_config(config_path)
        config = merge_with_default_config(config)

        # Ensure Agent config exists and set repository
        self.logger.assert_true('Agent' in config, "Agent not in config")
        self.logger.assert_true('agent_api' in config['Agent'] \
            and not config['Agent']['agent_api'], "Agent API not disabled")
        config['Agent']['repository'] = str(self.benchmark_dir.absolute())

        return config

    # =========================================================================
    # UNIFIED EVALUATION METHODS - MAXIMIZED CODE REUSE
    # =========================================================================

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

    def _prepare_evaluation(self, solutions: List[Dict]) -> Optional[Path]:
        """Prepare evaluation directory and files, return eval_dir or None if failed"""
        if not solutions:
            return None

        eval_dir = Path(self.agent.get_log_dir()) / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Save solutions
        solutions_file = eval_dir / "solutions.jsonl"
        write_jsonl(str(solutions_file), solutions)

        # Remove existing eval results
        for eval_results_file in eval_dir.glob("*.eval_results.json"):
            eval_results_file.unlink()

        return eval_dir

    def _run_eval_script(self, eval_script: Path, blocking: bool = True, timeout: int = 1800) -> Optional[str]:
        """Run evaluation script and return combined output"""
        env = os.environ.copy()
        cmd = ["bash", str(eval_script.absolute())]

        if blocking:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            output = f"{result.stdout}\n{result.stderr}\nEXIT_CODE:{result.returncode}"
            return output.strip()
        else:
            # Check if previous process is still running
            if self.eval_process and self.eval_process.poll() is None:
                self.logger.info("Evaluation already running, skipping")
                return None

            self.logger.info(f"Running eval script: {' '.join(cmd)}")
            self.eval_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=env
            )
            return None

    def _create_eval_script(self, eval_dir: Path, solutions_file: Path) -> Path:
        """Create compact evaluation script"""
        eval_script = eval_dir / "eval_script.sh"

        # Use absolute paths
        eval_dir = eval_dir.absolute()
        solutions_file = solutions_file.absolute()
        sanitized_file = str(solutions_file).replace('.jsonl', '-sanitized.jsonl')
        output_file = eval_dir / "eval_script.output.txt"

        script_content = f"""#!/bin/bash
cd "{eval_dir}" || exit 1
CMD=$(command -v evalplus >/dev/null && echo "evalplus" || echo "python -m evalplus")
echo "=== EVALPLUS: {self.dataset} | $(wc -l < '{solutions_file}') solutions ===" | tee {output_file}
$CMD.sanitize --samples "{solutions_file}" 2>&1 | tee -a {output_file} || true
EVAL_FILE=$([ -f "{sanitized_file}" ] && echo "{sanitized_file}" || echo "{solutions_file}")
$CMD.evaluate --dataset {self.dataset} --samples "$EVAL_FILE" | tee -a {output_file}
"""
        eval_script.write_text(script_content)
        eval_script.chmod(0o755)
        return eval_script

    def _parse_evaluation_scores(self, combined_output: str) -> Optional[Dict]:
        """Parse pass@k scores from EvalPlus output"""
        try:
            lines = combined_output.strip().split('\n')
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
        except Exception:
            return None

    def setup_problems(self, num_samples: Optional[int] = None,
                      task_ids: Optional[List[str]] = None) -> Dict:
        """Setup benchmark directory with problems"""
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

            # Make sure not to overwrite any existing problem code
            problem_file = problem_dir / f"{safe_id}.py"
            if not problem_file.exists():
                problem_file.write_text(problem["prompt"], encoding="utf-8")

            self.problems[task_id] = {"problem_dir": problem_dir, "safe_id": safe_id}

        self.logger.info(f"Setup {len(self.problems)} problems in {self.benchmark_dir}")
        return self.problems

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
        """Run EvalPlus evaluation on solutions (blocking)"""
        self.logger.info("Starting evaluation")

        try:
            # Prepare evaluation
            eval_dir = self._prepare_evaluation(solutions)
            if not eval_dir:
                self.logger.error("Failed to prepare evaluation - no solutions found")
                return {"error": "No solutions found"}

            solutions_file = eval_dir / "solutions.jsonl"

            # Verify solutions file exists
            if not solutions_file.exists():
                self.logger.error(f"Solutions file was not created: {solutions_file}")
                return {"error": f"Solutions file not created: {solutions_file}"}

            self.logger.info(f"Created solutions file: {solutions_file}")

            # Create and run script
            eval_script = self._create_eval_script(eval_dir, solutions_file)
            self.logger.info(f"Created evaluation script: {eval_script}")

            self.logger.info("Executing evaluation script...")
            combined_output = self._run_eval_script(eval_script)

            if combined_output is None:
                self.logger.error("Script execution returned None")
                return {"error": "evaluation script execution failed"}

            # Extract exit code
            exit_code_match = re.search(r'EXIT_CODE:(\d+)', combined_output)
            exit_code = int(exit_code_match.group(1)) if exit_code_match else 1
            self.logger.info(f"Evaluation completed with exit code: {exit_code}")

            if exit_code == 0:
                scores = self._parse_evaluation_scores(combined_output)
                if scores:
                    self.logger.info(f"Successfully parsed scores: {scores}")
                    return {"output": combined_output, "scores": scores}
                else:
                    self.logger.warning("Evaluation completed but no scores found")
                    return {"output": combined_output}
            else:
                self.logger.error(f"Evaluation failed with exit code {exit_code}")
                return {"error": "evaluation failed", "output": combined_output, "exit_code": exit_code}

        except subprocess.TimeoutExpired:
            self.logger.error("Evaluation timeout exceeded")
            return {"error": "evaluation timeout"}
        except Exception as e:
            self.logger.error(f"Evaluation exception: {e}")
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

    # =========================================================================
    # AGENT EXECUTION WITH CALLBACK-BASED PERIODIC EVALUATION
    # =========================================================================

    def run_agent_with_eval_checks(self, max_iterations: int, stop_on_error: bool) -> Dict:
        """Run agent with periodic evaluation using minimal callback system"""
        agent_config = self.config.get('Agent', {})

        self.agent = Agent(
            name=agent_config['name'],
            role=agent_config['role'],
            repository=str(self.benchmark_dir),
            config=self.config
        )

        # Set up periodic evaluation callback if enabled
        if self.eval_interval:
            self.agent.set_callback(self._evaluation_callback)
            self.logger.info(f"Enabled periodic evaluation every {self.eval_interval}s using callback")

        # Run agent normally - callback will handle periodic evaluation
        return self.agent.run_loop(max_iterations=max_iterations, stop_on_error=stop_on_error)

    def _evaluation_callback(self, agent, iteration):
        """Callback executed for evaluation at the end of each iteration"""
        current_time = time.time()
        if current_time - self.last_eval_time < self.eval_interval:
            return

        solutions = self.extract_solutions()
        if not any(sol["solution"].strip() for sol in solutions):
            return

        try:
            eval_dir = self._prepare_evaluation(solutions)
            if not eval_dir:
                return

            solutions_file = eval_dir / "solutions.jsonl"
            eval_script = self._create_eval_script(eval_dir, solutions_file)
            self._run_eval_script(eval_script, blocking=False)

            self.logger.info(f"Started background evaluation at iteration {iteration}")
            self.last_eval_time = current_time

        except Exception as e:
            self.logger.error(f"Failed to start background evaluation: {e}")
            self.logger.error(traceback.format_exc())

    # =========================================================================
    # WORKFLOW AND RESULT MANAGEMENT
    # =========================================================================

    def save_results(self, agent_results: Dict, evaluation_results: Dict) -> Path:
        """Save benchmark results"""
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
                "total_actions": agent_results.get('execution_stats', {}).get('actions_executed', 0),
                "final_state": agent_results.get('agent_info', {}).get('current_state', 'unknown'),
                "evaluation_success": "scores" in evaluation_results,
                "scores": evaluation_results.get("scores")
            }
        }

        results_file = eval_dir / "benchmark.results.json"
        try:
            results_file.write_text(json.dumps(results, indent=4, default=str))
            return results_file
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return None

    def run_complete_benchmark(self, max_iterations: int = 10, stop_on_error: bool = False,
                              num_samples: Optional[int] = None, task_ids: Optional[List[str]] = None,
                              run_evaluation: bool = True, eval_interval: Optional[int] = None) -> Tuple[Dict, Path]:
        """Run complete benchmark pipeline with callback-based evaluation"""
        try:
            # Setup problems
            self.setup_problems(num_samples, task_ids)

            # Enable periodic evaluation if requested
            if eval_interval and run_evaluation:
                self.eval_interval = eval_interval
                self.last_eval_time = 0
                self.logger.info(f"Enabled periodic evaluation every {eval_interval}s using callbacks")

            # Run agent with callback-based evaluation
            agent_results = self.run_agent_with_eval_checks(max_iterations, stop_on_error)

            # Run final evaluation
            evaluation_results = {}
            if run_evaluation:
                self.logger.info("Running final evaluation...")
                solutions = self.extract_solutions()
                evaluation_results = self.run_evaluation(solutions)

            # Save results
            results_file = self.save_results(agent_results, evaluation_results)

            return {
                "agent_results": agent_results,
                "evaluation_results": evaluation_results,
                "summary": {
                    "problems": len(self.problems),
                    "actions": agent_results.get('execution_stats', {}).get('actions_executed', 0),
                    "final_state": agent_results.get('agent_info', {}).get('current_state'),
                    "scores": evaluation_results.get("scores")
                }
            }, results_file

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
    parser.add_argument("--num-samples", type=int, help="Number of samples to include")
    parser.add_argument("--task-ids", nargs="+", help="Specific task IDs to include")
    parser.add_argument("--max-iterations", type=int, default=0, help="Iterations for agent to run")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop agent if exception thrown")
    parser.add_argument("--no-evaluation", action="store_true", help="Skip evaluation")
    parser.add_argument("--eval-interval", type=int, default=1800, help="Periodic evaluation interval in seconds")

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
        benchmark.logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
