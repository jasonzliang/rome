# multi_agent_benchmark.py
import multiprocessing as mp
import os
import signal
import sys
import time
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add the parent directory to sys.path to import from the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_humaneval import EvalPlusBenchmark


def run_single_agent_worker(args: Tuple[str, str, Dict, int, bool, int]) -> Dict:
    """Worker function to run a single agent in a separate process - no evaluation."""
    benchmark_dir, config_path, config_overrides, max_iterations, stop_on_error, agent_id = args

    try:
        # Create benchmark instance using shared benchmark directory
        benchmark = EvalPlusBenchmark(benchmark_dir, config_path, "humaneval")

        if config_overrides:
            def deep_update(base_dict, update_dict):
                for key, value in update_dict.items():
                    if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                        deep_update(base_dict[key], value)
                    else:
                        base_dict[key] = value
            deep_update(benchmark.config, config_overrides)

        # Run only the agent (no setup problems, no evaluation)
        agent_results = benchmark.run_agent(max_iterations, stop_on_error)

        return {
            "success": True,
            "agent_id": agent_id,
            "agent_results": agent_results,
            "process_id": os.getpid()
        }

    except Exception as e:
        return {
            "success": False,
            "agent_id": agent_id,
            "error": str(e),
            "process_id": os.getpid()
        }


class EvalPlusBenchmarkMulti(EvalPlusBenchmark):
    """Multi-agent benchmark runner with shared directory and centralized evaluation."""

    def __init__(self, benchmark_dir: str, config_path: str, dataset: str = "humaneval",
                 num_agents: int = 2, max_workers: Optional[int] = None, eval_interval: Optional[int] = None):
        super().__init__(benchmark_dir, config_path, dataset)

        self.num_agents = num_agents
        self.max_workers = max_workers or min(num_agents, mp.cpu_count())
        self.config_path = config_path
        self.eval_interval = eval_interval
        self.executor = None

        # Multi-agent evaluation thread
        self.multi_eval_thread = None
        self.multi_eval_stop_event = threading.Event()
        self.agent_results = {}  # Store results from all agents

        eval_msg = f", eval every {eval_interval}s" if eval_interval else ", no periodic eval"
        self.logger.info(f"Initialized EvalPlusBenchmarkMulti with {num_agents} agents using shared directory {benchmark_dir}, {self.max_workers} workers{eval_msg}")

    def _multi_agent_evaluation_worker(self):
        """Single evaluation thread that evaluates all agents at regular intervals."""
        while not self.multi_eval_stop_event.wait(self.eval_interval):
            try:
                self.logger.info("Running periodic evaluation for all agents...")

                # Extract solutions and run evaluation using shared directory
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
                            solution_count = len([s for s in solutions if s["solution"].strip()])
                            self.logger.info(f"Multi-agent eval ({self.num_agents} agents, {solution_count} solutions) - {', '.join(score_summary)}")

            except Exception as e:
                self.logger.error(f"Multi-agent periodic evaluation failed: {e}")

    def start_multi_agent_evaluation(self):
        """Start centralized evaluation thread for all agents."""
        if not self.eval_interval or self.eval_interval <= 0:
            return

        self.multi_eval_stop_event.clear()
        self.multi_eval_thread = threading.Thread(
            target=self._multi_agent_evaluation_worker,
            name="MultiAgentEvalThread",
            daemon=True
        )
        self.multi_eval_thread.start()
        self.logger.info(f"Started multi-agent evaluation thread (interval: {self.eval_interval}s)")

    def stop_multi_agent_evaluation(self):
        """Stop centralized evaluation thread."""
        if self.multi_eval_thread and self.multi_eval_thread.is_alive():
            self.multi_eval_stop_event.set()
            self.logger.info("Stopped multi-agent evaluation thread")

    def run_agents_parallel(self, max_iterations: int = 10, stop_on_error: bool = False,
                          config_overrides: Optional[Dict] = None) -> List[Dict]:
        """Run multiple agents in parallel using shared benchmark directory."""

        # Prepare agent arguments - all use same benchmark_dir
        agent_args = [
            (str(self.benchmark_dir), self.config_path, config_overrides or {}, max_iterations, stop_on_error, i)
            for i in range(self.num_agents)
        ]

        results = []
        self.logger.info(f"Starting {len(agent_args)} agents in shared directory")

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            self.executor = executor

            try:
                future_to_agent = {
                    executor.submit(run_single_agent_worker, args): args[5]  # agent_id
                    for args in agent_args
                }

                for future in as_completed(future_to_agent):
                    agent_id = future_to_agent[future]

                    try:
                        result = future.result(timeout=3600)
                        results.append(result)

                        # Store agent results for potential use
                        if result["success"]:
                            self.agent_results[agent_id] = result["agent_results"]

                        status = "completed" if result["success"] else f"failed: {result['error']}"
                        self.logger.info(f"Agent {agent_id:03d} {status}")

                    except Exception as e:
                        results.append({
                            "success": False,
                            "agent_id": agent_id,
                            "error": f"Execution failed: {str(e)}",
                            "process_id": None
                        })
                        self.logger.error(f"Agent {agent_id:03d} execution failed: {e}")

            except KeyboardInterrupt:
                self.logger.info("Interrupt received, shutting down agents...")
                self._shutdown_all_agents(executor)
                raise
            finally:
                self.executor = None

        success_count = sum(1 for r in results if r["success"])
        self.logger.info(f"Completed: {success_count}/{len(results)} agents successful")

        return results

    def _shutdown_all_agents(self, executor: ProcessPoolExecutor):
        """Gracefully shutdown all agent processes."""
        # Cancel pending futures
        for future in executor._futures:
            future.cancel()

        # Terminate processes
        for process in executor._processes.values():
            if process.is_alive():
                try:
                    os.kill(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        time.sleep(2)

        # Force kill remaining processes
        for process in executor._processes.values():
            if process.is_alive():
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def run_final_evaluation(self, results: List[Dict]) -> List[Dict]:
        """Run final evaluation using shared directory."""
        self.logger.info("Running final evaluation for all agents...")

        try:
            # Extract solutions and run evaluation using parent class methods
            solutions = self.extract_solutions()
            evaluation_results = self.run_evaluation(solutions)

            # Add evaluation results to all successful agent results
            for result in results:
                if result["success"]:
                    result["evaluation_results"] = evaluation_results
                    result["evaluation_success"] = "scores" in evaluation_results
                    result["scores"] = evaluation_results.get("scores")

            self.logger.info("Final evaluation completed for all agents")

        except Exception as e:
            self.logger.error(f"Final evaluation failed: {e}")
            for result in results:
                if result["success"]:
                    result["evaluation_error"] = str(e)

        return results

    def run_complete_multi_agent_benchmark(self, max_iterations: int = 10,
                                         stop_on_error: bool = False,
                                         num_samples: Optional[int] = None,
                                         task_ids: Optional[List[str]] = None,
                                         config_overrides: Optional[Dict] = None) -> List[Dict]:
        """Run the complete multi-agent benchmark with shared directory."""
        try:
            # 1. Setup problems once in shared directory
            self.setup_problems(num_samples, task_ids)

            # 2. Start centralized evaluation thread
            self.start_multi_agent_evaluation()

            # 3. Run all agents in parallel using shared directory
            results = self.run_agents_parallel(
                max_iterations=max_iterations,
                stop_on_error=stop_on_error,
                config_overrides=config_overrides
            )

            # 4. Stop centralized evaluation thread
            self.stop_multi_agent_evaluation()

            # 5. Run final evaluation
            evaluated_results = self.run_final_evaluation(results)

            return evaluated_results

        except KeyboardInterrupt:
            self.logger.info("Benchmark interrupted by user")
            self.stop_multi_agent_evaluation()
            raise
        except Exception as e:
            self.logger.error(f"Multi-agent benchmark failed: {e}")
            self.stop_multi_agent_evaluation()
            raise

    def cleanup(self):
        """Clean up resources."""
        self.stop_multi_agent_evaluation()
        if self.executor:
            self._shutdown_all_agents(self.executor)

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def main():
    """Main function for EvalPlusBenchmarkMulti"""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-agent benchmark on HumanEval+ dataset")
    parser.add_argument("benchmark_dir", help="Shared directory for all agents")
    parser.add_argument("config_file", help="Path to agent configuration YAML file")
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--num-agents", type=int, default=2, help="Number of agent instances")
    parser.add_argument("--max-workers", type=int, help="Maximum concurrent workers")
    parser.add_argument("--num-samples", type=int, help="Number of samples per agent")
    parser.add_argument("--task-ids", nargs="+", help="Specific task IDs to include")
    parser.add_argument("--eval-interval", type=int, help="Evaluation interval in seconds (0 to disable)")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max iterations per agent")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop agents on errors")

    args = parser.parse_args()

    benchmark = EvalPlusBenchmarkMulti(
        benchmark_dir=args.benchmark_dir,
        config_path=args.config_file,
        dataset=args.dataset,
        num_agents=args.num_agents,
        max_workers=args.max_workers,
        eval_interval=args.eval_interval
    )

    try:
        results = benchmark.run_complete_multi_agent_benchmark(
            max_iterations=args.max_iterations,
            stop_on_error=args.stop_on_error,
            num_samples=args.num_samples,
            task_ids=args.task_ids
        )

        success_count = sum(1 for r in results if r["success"])
        print(f"\nCompleted {success_count}/{len(results)} agents successfully")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    main()
