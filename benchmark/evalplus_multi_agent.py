# evalplus_multi_agent.py - Compact multi-agent benchmark with asyncio
import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalplus_single_agent import EvalPlusBenchmark
from rome.multi_agent import MultiAgent


class EvalPlusMultiAgentBenchmark(EvalPlusBenchmark):
    """Compact multi-agent benchmark with asyncio periodic evaluation"""

    def __init__(self, benchmark_dir: str, config_path: str, agents_config_path: Optional[str] = None,
                 dataset: str = "humaneval", eval_interval: Optional[int] = None):
        super().__init__(benchmark_dir, config_path, dataset)

        # Resolve agents config path with priority: parse arg > config file > error
        self.agents_config_path = self._resolve_agents_config_path(agents_config_path)

        self.eval_interval = eval_interval
        self.multi_agent = None
        self.eval_task = None
        self.eval_stop_event = asyncio.Event()

        self.logger.info(f"Multi-agent benchmark initialized: {self.agents_config_path}")

    def _resolve_agents_config_path(self, agents_config_path: Optional[str]) -> str:
        """Resolve agents config path with priority: parse arg > config file > error"""
        # Priority 1: Command line argument
        if agents_config_path:
            if not Path(agents_config_path).exists():
                raise FileNotFoundError(f"Agents config file not found: {agents_config_path}")
            self.logger.info(f"Using agents config from command line: {agents_config_path}")
            return agents_config_path

        # Priority 2: Config file
        multi_agent_config = self.config.get('MultiAgent', {})
        config_agents_path = multi_agent_config.get('agent_role_json')

        if config_agents_path:
            # Handle relative paths from config file directory
            config_dir = Path(self.config.get('_config_file_path', '.')).parent
            resolved_path = config_dir / config_agents_path if not Path(config_agents_path).is_absolute() else Path(config_agents_path)

            if resolved_path.exists():
                self.logger.info(f"Using agents config from config file: {resolved_path}")
                return str(resolved_path)
            else:
                self.logger.error(f"Agents config path from config file not found: {resolved_path}")

        # Priority 3: Error if neither provided
        raise ValueError(
            "Agents configuration path required. Provide either:\n"
            "  1. Command line argument: --agents-config path/to/agents.json\n"
            "  2. Config file setting: MultiAgent.agent_role_json"
        )

    async def _run_evaluation_async(self, solutions: List[Dict]) -> Optional[Dict]:
        """Run evaluation asynchronously using subprocess"""
        if not solutions:
            return None

        try:
            eval_dir = self._prepare_evaluation(solutions)
            if not eval_dir:
                return None

            solutions_file = eval_dir / "solutions.jsonl"
            eval_script = self._create_eval_script(eval_dir, solutions_file)

            # Run evaluation script asynchronously
            process = await asyncio.create_subprocess_shell(
                f"bash {eval_script.absolute()}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # cwd=eval_dir
            )

            stdout, stderr = await process.communicate()
            combined_output = f"{stdout.decode()}\n{stderr.decode()}\nEXIT_CODE:{process.returncode}"

            if process.returncode == 0:
                scores = self._parse_evaluation_scores(combined_output)
                return {"output": combined_output, "scores": scores} if scores else {"output": combined_output}
            else:
                return {"error": "evaluation failed", "output": combined_output, "exit_code": process.returncode}

        except Exception as e:
            self.logger.error(f"Async evaluation failed: {e}")
            return {"error": str(e)}

    async def _periodic_evaluation_worker(self):
        """Asyncio-based periodic evaluation worker"""
        last_solution_count = 0

        while not self.eval_stop_event.is_set():
            try:
                await asyncio.sleep(self.eval_interval)

                solutions = self.extract_solutions()
                solution_count = len([s for s in solutions if s["solution"].strip()])

                # Skip if no new solutions
                if solution_count == 0 or solution_count == last_solution_count:
                    continue

                last_solution_count = solution_count
                results = await self._run_evaluation_async(solutions)

                if results and "scores" in results:
                    scores = [f"{k}: {v.get('pass@1', 0):.3f}"
                             for k, v in results["scores"].items() if "pass@1" in v]
                    if scores:
                        agent_count = len(self.multi_agent.agents_config) if self.multi_agent else 0
                        self.logger.info(f"Async eval ({agent_count} agents, {solution_count} solutions) - {', '.join(scores)}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Periodic evaluation failed: {e}")

    def _start_eval_task(self):
        """Start asyncio evaluation task"""
        if not self.eval_interval or self.eval_interval <= 0:
            return
        self.eval_stop_event.clear()
        self.eval_task = asyncio.create_task(self._periodic_evaluation_worker())
        self.logger.info(f"Started async evaluation task (interval: {self.eval_interval}s)")

    async def _stop_eval_task(self):
        """Stop asyncio evaluation task"""
        if self.eval_task and not self.eval_task.done():
            self.eval_stop_event.set()
            self.eval_task.cancel()
            try:
                await self.eval_task
            except asyncio.CancelledError:
                pass

    def run_agents_parallel(self, max_iterations: int, stop_on_error: bool) -> List[Dict]:
        """Run agents using MultiAgent class"""
        self.multi_agent = MultiAgent(self.agents_config_path, str(self.benchmark_dir), self.config)

        try:
            results = self.multi_agent.run_loop(max_iterations=max_iterations, timeout_per_agent=0)

            # Convert to expected format
            agent_results = []
            for name, result in results['agent_results'].items():
                if 'error' in result:
                    agent_results.append({
                        "success": False, "agent_name": name,
                        "error": result['error']['error_message'],
                        "error_type": result['error']['error_type']
                    })
                else:
                    agent_results.append({
                        "success": True, "agent_name": name, "agent_results": result
                    })

            success_count = sum(1 for r in agent_results if r["success"])
            self.logger.info(f"Completed: {success_count}/{len(agent_results)} agents")
            return agent_results

        finally:
            if self.multi_agent:
                self.multi_agent.shutdown()

    async def run_final_evaluation_async(self, agent_results: List[Dict]) -> Dict:
        """Run final evaluation asynchronously"""
        try:
            solutions = self.extract_solutions()
            evaluation_results = await self._run_evaluation_async(solutions)
            if not evaluation_results:
                evaluation_results = {}

            for result in agent_results:
                if result["success"]:
                    result.update({
                        "evaluation_results": evaluation_results,
                        "evaluation_success": "scores" in evaluation_results,
                        "scores": evaluation_results.get("scores")
                    })
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Final evaluation failed: {e}")
            return {"error": str(e)}

    def save_results(self, agent_results: List[Dict], evaluation_results: Dict) -> Path:
        """Save compact results"""
        results = {
            "benchmark_type": "multi_agent", "dataset": self.dataset,
            "benchmark_dir": str(self.benchmark_dir), "agents_config": self.agents_config_path,
            "problems_count": len(self.problems), "total_agents": len(agent_results),
            "successful_agents": sum(1 for r in agent_results if r["success"]),
            "evaluation_results": evaluation_results, "agent_results": agent_results,
            "scores": evaluation_results.get("scores")
        }

        results_file = Path(self.benchmark_dir) / "__rome__" / "multi_agent_results.json"
        results_file.parent.mkdir(exist_ok=True)
        results_file.write_text(json.dumps(results, indent=2, default=str))
        return results_file

    def print_summary(self, results: Dict):
        """Print compact summary"""
        scores = results.get("scores", {})
        agent_results = results.get("agent_results", [])

        self.logger.info("="*60)
        self.logger.info(f"Multi-Agent {self.dataset.upper()} | Problems: {len(self.problems)} | "
                        f"Agents: {results.get('successful_agents', 0)}/{results.get('total_agents', 0)}")

        if scores:
            score_strs = [f"{k.replace('_', '+').title()}: {v.get('pass@1', 0):.3f}"
                         for k, v in scores.items() if isinstance(v, dict) and 'pass@1' in v]
            if score_strs:
                self.logger.info(f"Scores: {' | '.join(score_strs)}")

        # Agent status summary
        if agent_results:
            success_agents = [r["agent_name"] for r in agent_results if r["success"]]
            failed_agents = [f"{r['agent_name']}({r.get('error_type', 'error')})"
                           for r in agent_results if not r["success"]]
            if success_agents:
                self.logger.info(f"Success: {', '.join(success_agents)}")
            if failed_agents:
                self.logger.info(f"Failed: {', '.join(failed_agents)}")
        self.logger.info("="*60)

    async def run_complete_benchmark_async(self, max_iterations: int = 10, stop_on_error: bool = False,
                                          num_samples: Optional[int] = None, task_ids: Optional[List[str]] = None,
                                          run_evaluation: bool = True) -> Tuple[Dict, Path]:
        """Complete benchmark pipeline with asyncio"""
        try:
            # Setup problems
            self.setup_problems(num_samples, task_ids)

            # Start async evaluation if enabled
            if run_evaluation and self.eval_interval:
                self._start_eval_task()

            # Run agents (still synchronous as MultiAgent doesn't support async)
            agent_results = self.run_agents_parallel(max_iterations, stop_on_error)

            # Stop evaluation task
            await self._stop_eval_task()

            # Run final evaluation asynchronously
            evaluation_results = await self.run_final_evaluation_async(agent_results) if run_evaluation else {}
            results_file = self.save_results(agent_results, evaluation_results)

            return {
                "agent_results": agent_results, "evaluation_results": evaluation_results,
                "scores": evaluation_results.get("scores"), "total_agents": len(agent_results),
                "successful_agents": sum(1 for r in agent_results if r["success"])
            }, results_file

        except KeyboardInterrupt:
            self.logger.info("Benchmark interrupted")
            await self._stop_eval_task()
            raise

    def run_complete_benchmark(self, max_iterations: int = 10, stop_on_error: bool = False,
                              num_samples: Optional[int] = None, task_ids: Optional[List[str]] = None,
                              run_evaluation: bool = True) -> Tuple[Dict, Path]:
        """Sync wrapper for async benchmark"""
        return asyncio.run(self.run_complete_benchmark_async(
            max_iterations, stop_on_error, num_samples, task_ids, run_evaluation
        ))


def main():
    parser = argparse.ArgumentParser(description="Multi-agent EvalPlus benchmark with AsyncIO")
    parser.add_argument("benchmark_dir", help="Shared benchmark directory")
    parser.add_argument("config_file", help="Agent configuration YAML")
    parser.add_argument("--agents-config", help="Agents configuration JSON (overrides config file)")
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--num-samples", type=int, help="Number of samples")
    parser.add_argument("--task-ids", nargs="+", help="Specific task IDs")
    parser.add_argument("--eval-interval", type=int, help="Evaluation interval (seconds)")
    parser.add_argument("--max-iterations", type=int, default=0, help="Max iterations per agent")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on errors")
    parser.add_argument("--no-evaluation", action="store_true", help="Skip evaluation")

    args = parser.parse_args()

    try:
        benchmark = EvalPlusMultiAgentBenchmark(
            args.benchmark_dir, args.config_file, args.agents_config,
            args.dataset, args.eval_interval
        )

        results, results_file = benchmark.run_complete_benchmark(
            args.max_iterations, args.stop_on_error, args.num_samples,
            args.task_ids, not args.no_evaluation
        )
        benchmark.print_summary(results)
        benchmark.logger.info(f"Results: {results_file}")

    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 1

    return 0


if __name__ == "__main__":
    main()
