# benchmark_humaneval.py
# Benchmark agent performance on HumanEval+ dataset using EvalPlus
import argparse
import json
import os
import shutil
import subprocess
import sys
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

    def __init__(self, config_path: str, benchmark_dir: str, dataset: str = "humaneval"):
        self.logger = get_logger()
        self.benchmark_dir = Path(benchmark_dir)
        self.dataset = dataset.lower()
        self.config = self._load_config(config_path)
        self.agent = None
        self.problems = {}

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
        datasets = {
            'humaneval': get_human_eval_plus,
            'mbpp': get_mbpp_plus
        }
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
        # Clear and create benchmark directory
        if self.benchmark_dir.exists():
            shutil.rmtree(self.benchmark_dir)
        self.benchmark_dir.mkdir(parents=True)

        # Get and filter problems
        problems = self._get_dataset()
        if task_ids:
            problems = {tid: problems[tid] for tid in task_ids if tid in problems}
        if num_samples and num_samples < len(problems):
            problems = dict(list(problems.items())[:num_samples])

        # Create problem directories
        self.problems = {}
        for task_id, problem in problems.items():
            safe_id = self._make_safe_task_id(task_id)
            problem_dir = self.benchmark_dir / safe_id
            problem_dir.mkdir(exist_ok=True)

            # Write problem file and metadata
            (problem_dir / f"{safe_id}.py").write_text(problem["prompt"], encoding="utf-8")
            (problem_dir / "metadata.json").write_text(
                json.dumps({
                    "task_id": task_id,
                    "entry_point": problem["entry_point"],
                    "canonical_solution": problem.get("canonical_solution", ""),
                    "base_input": problem.get("base_input", []),
                    "plus_input": problem.get("plus_input", [])
                }, indent=4), encoding="utf-8"
            )

            self.problems[task_id] = {
                "problem_dir": problem_dir,
                "safe_id": safe_id
            }

        self.logger.info(f"Setup {len(self.problems)} problems in {self.benchmark_dir}")
        return self.problems

    def run_agent(self, max_iterations: int = 10, stop_on_error: bool = False) -> Dict:
        """Run agent on benchmark problems"""
        agent_config = self.config.get('Agent', {})

        self.agent = Agent(
            name=agent_config.get('name', 'EvalPlusBenchmark'),
            role=agent_config.get('role', 'You are an expert Python developer tasked with implementing functions to pass all given test cases.'),
            config=self.config
        )

        # Export config using the agent's method instead of manual YAML saving
        try:
            self.agent.export_config()
            self.logger.info("Agent configuration exported successfully")
        except Exception as e:
            self.logger.error(f"Could not export agent configuration: {e}")

        # Draw FSM graph
        try:
            self.agent.draw_fsm_graph()
        except Exception as e:
            self.logger.error(f"Could not generate FSM graph: {e}")

        # Run agent
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
                self.logger.error(f"No solution file found for {task_id}")

        self.logger.info(f"Extracted {len(solutions)} solutions")
        return solutions

    def run_evaluation(self, solutions: List[Dict]) -> Dict:
        """Run EvalPlus evaluation on solutions"""
        if not solutions:
            return {"error": "No solutions found"}

        eval_dir = Path(self.agent.get_log_dir()) / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Save solutions and run sanitization
        solutions_file = eval_dir / "solutions.jsonl"
        write_jsonl(str(solutions_file), solutions)

        sanitized_result = self._run_evalplus_command(
            ["python", "-m", "evalplus.sanitize", "--samples", str(solutions_file)],
            eval_dir, "sanitization"
        )

        # Check if sanitization failed
        if isinstance(sanitized_result, dict) and "error" in sanitized_result:
            return sanitized_result

        # Get the sanitized file path
        sanitized_file = sanitized_result if isinstance(sanitized_result, Path) else None
        samples_file = sanitized_file if sanitized_file and sanitized_file.exists() else solutions_file

        # Run evaluation
        eval_result = self._run_evalplus_command(
            ["python", "-m", "evalplus.evaluate", "--dataset", self.dataset,
             "--samples", str(samples_file)],
            eval_dir, "evaluation", timeout=1800
        )

        # Check if evaluation failed
        if "error" in eval_result:
            return eval_result

        # Parse scores from output
        scores = self._parse_evaluation_scores(eval_result.get("stdout", ""))
        if scores:
            eval_result["scores"] = scores

        return eval_result

    def _run_evalplus_command(self, cmd: List[str], cwd: Path, operation: str,
                             timeout: int = 300):
        """Run EvalPlus command with error handling"""
        try:
            result = subprocess.run(
                cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout
            )

            if result.returncode == 0:
                self.logger.info(f"EvalPlus {operation} completed successfully")
                # For sanitize, check if output file exists and return path or success dict
                if operation == "sanitization":
                    sanitized_file = cwd / "solutions-sanitized.jsonl"
                    if sanitized_file.exists():
                        return sanitized_file
                    else:
                        # Return success but note no sanitized file was created
                        return {
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "return_code": result.returncode,
                            "note": "No sanitized file created"
                        }

                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                }
            else:
                self.logger.error(f"EvalPlus {operation} failed: {result.stderr}")
                return {"error": f"{operation} failed", "stderr": result.stderr}

        except subprocess.TimeoutExpired:
            self.logger.error(f"EvalPlus {operation} timed out")
            return {"error": f"{operation} timed out", "timeout": timeout}
        except Exception as e:
            self.logger.error(f"EvalPlus {operation} error: {e}")
            return {"error": str(e)}

    def _parse_evaluation_scores(self, stdout: str) -> Optional[Dict]:
        """Parse pass@k scores from EvalPlus output"""
        try:
            lines = stdout.strip().split('\n')
            scores = {}

            for i, line in enumerate(lines):
                if line.strip() == "Base" and i + 1 < len(lines):
                    try:
                        scores["base"] = eval(lines[i + 1])
                    except:
                        pass
                elif line.strip() == "Base + Extra" and i + 1 < len(lines):
                    try:
                        scores["base_plus_extra"] = eval(lines[i + 1])
                    except:
                        pass

            return scores if scores else None
        except Exception as e:
            self.logger.error(f"Could not parse evaluation scores: {e}")
            return None

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
            self.logger.info(f"Results saved to: {results_file}")
            return results_file
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return None

    def run_complete_benchmark(self, max_iterations: int = 10, stop_on_error: bool = False,
                              num_samples: Optional[int] = None, task_ids: Optional[List[str]] = None,
                              run_evaluation: bool = True) -> Tuple[Dict, Path]:
        """Run complete benchmark pipeline"""
        try:
            # Setup problems
            self.setup_problems(num_samples, task_ids)

            # Run agent
            agent_results = self.run_agent(max_iterations, stop_on_error)

            # Run evaluation
            evaluation_results = {}
            if run_evaluation:
                solutions = self.extract_solutions()
                evaluation_results = self.run_evaluation(solutions)

            # Save and return results
            results_file = self.save_results(agent_results, evaluation_results)

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

        finally:
            if self.agent:
                self.agent.shutdown()

    def print_summary(self, results: Dict):
        """Print benchmark summary"""
        summary = results.get("summary", {})
        scores = summary.get("scores", {})

        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Dataset: {self.dataset}")
        print(f"Problems: {summary.get('problems', 0)}")
        print(f"Actions: {summary.get('actions', 0)}")
        print(f"Final State: {summary.get('final_state', 'unknown')}")

        if scores:
            print("\nEVALUATION RESULTS:")
            for score_type, metrics in scores.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        print(f"{score_type.replace('_', ' ').title()} {metric}: {value}")

        print("="*60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Benchmark agent on HumanEval+ dataset")
    parser.add_argument("config_file", help="Path to agent configuration YAML file")
    parser.add_argument("benchmark_dir", help="Path to benchmark directory (will be created/cleared)")
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--num-samples", type=int, help="Number of samples to include")
    parser.add_argument("--task-ids", nargs="+", help="Specific task IDs to include")
    parser.add_argument("--max-iterations", type=int, default=4000)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--no-evaluation", action="store_false", dest="run_evaluation")

    args = parser.parse_args()

    # Setup logging
    get_logger().configure({
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "console": True
    })

    # Run benchmark
    benchmark = EvalPlusBenchmark(args.config_file, args.benchmark_dir, args.dataset)

    try:
        results, results_file = benchmark.run_complete_benchmark(
            max_iterations=args.max_iterations,
            stop_on_error=args.stop_on_error,
            num_samples=args.num_samples,
            task_ids=args.task_ids,
            run_evaluation=args.run_evaluation
        )

        benchmark.print_summary(results)
        if results_file:
            print(f"\nDetailed results: {results_file}")

    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
