# Enhanced evalplus_evaluator.py with update_plot2 function
import asyncio
import json
import re
import os
import signal
import sys
import subprocess
import traceback
import threading
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
from datetime import datetime
import glob

import numpy as np
import matplotlib; matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

try:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
except ImportError:
    raise ImportError("EvalPlus not found. Install with: pip install evalplus")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rome.logger import get_logger
from rome.config import LOG_DIR_NAME, EVAL_DIR_NAME, EVAL_RESULTS_NAME

class EvalplusEvaluator:
    """Enhanced evaluator with score tracking and visualization"""

    DATASETS = {'humaneval': get_human_eval_plus, 'mbpp': get_mbpp_plus}

    def __init__(self,
        benchmark_dir: Union[str, Path],
        dataset: str = "humaneval"):

        if dataset not in self.DATASETS:
            raise ValueError(f"Dataset must be one of {list(self.DATASETS.keys())}")

        self.dataset = dataset
        self.benchmark_dir = Path(benchmark_dir)
        self.log_dir = self.benchmark_dir / LOG_DIR_NAME
        self.eval_dir = self.log_dir / EVAL_DIR_NAME
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger()
        self.problems = {}
        self.process = None

        # Score tracking - using list for simpler handling
        self.start_time = None
        self.scores = []  # List of [timestamp, elapsed_min, base_score, extra_score]
        self.scores_file = self.eval_dir / "score_history.json"
        self.plot_time = self.eval_dir / "score_time_plot.png"
        self.plot_iteration = self.eval_dir / "score_iteration_plot.png"
        self._load_scores()

    def _load_scores(self):
        """Load scores from file"""
        try:
            if self.scores_file.exists():
                data = json.loads(self.scores_file.read_text())
                self.scores = data.get('scores', [])
                self.start_time = data.get('start_time')
                if self.scores:
                    self.logger.info(f"Loaded {len(self.scores)} score entries")
        except Exception as e:
            self.logger.error(f"Load scores failed: {e}")
            self.scores = []

    def _save_and_plot_scores(self, scores: Optional[Dict], timestamp: float = None):
        """Record scores and update plot in one efficient operation"""
        if not scores:
            return

        # Extract and record scores
        base = scores.get('base', {}).get('pass@1', 0.0)
        extra = scores.get('base_plus_extra', {}).get('pass@1', 0.0)

        if self.start_time is None:
            self.start_time = timestamp or time.time()

        current_timestamp = timestamp or time.time()
        elapsed_min = (current_timestamp - self.start_time) / 60

        # Append to list
        self.scores.append([current_timestamp, elapsed_min, base, extra])

        # Save data
        self.scores_file.write_text(json.dumps({
                'start_time': self.start_time,
                'scores': self.scores,
                'dataset': self.dataset
            }))

        # Create plots
        self._plot_time()
        self._plot_iteration()
        self.logger.info(f"Scores at t={elapsed_min:.1f}m: base={base:.3f}, extra={extra:.3f}")

    def _plot_time(self):
        """Create/update plot efficiently with latest scores in legend"""
        if not self.scores:
            return

        try:
            # Extract data from list (already sorted by timestamp)
            times = [entry[1] for entry in self.scores]  # elapsed_min
            base_scores = [entry[2] for entry in self.scores]  # base_score
            extra_scores = [entry[3] for entry in self.scores]  # extra_score

            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot data with latest scores in labels
            ax.plot(times, base_scores, 'b-o',
                    label=f'Base ({base_scores[-1]:.3f} | {max(base_scores):.3f})',
                    linewidth=2, markersize=4)
            ax.plot(times, extra_scores, 'r-s',
                    label=f'Base+Extra ({extra_scores[-1]:.3f} | {max(extra_scores):.3f})',
                    linewidth=2, markersize=4)

            # Format
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Pass@1 Score')
            ax.set_title(f'{self.dataset.upper()} Scores Over Time')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

            # Save
            plt.tight_layout()
            plt.savefig(self.plot_time, dpi=200, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Scores vs time elapsed plot failed: {e}")

    def _plot_iteration(self):
        """Compact plot: total iterations vs scores with completion percentage and latest scores in legend"""
        if not self.scores:
            return

        try:
            agent_data = {}
            for file_path in self.log_dir.glob("agent_*.summary_history.json"):
                try:
                    data = json.loads(file_path.read_text())
                    entries = [(e['iteration'], e['epoch_time'],
                               e.get('summary', {}).get('repository_progress', {}).get('completion_percentage', 0))
                              for e in data.get('summary_history', [])
                              if 'iteration' in e and 'epoch_time' in e]
                    if entries:
                        agent_data[file_path.stem] = entries
                except:
                    continue

            if not agent_data:
                return

            # Calculate total iterations and completion for each score
            total_iters = []
            completion_fractions = []
            for score_entry in self.scores:
                target_time = score_entry[0]
                total_iter = 0.0
                total_completion = 0.0
                agent_count = 0

                for entries in agent_data.values():
                    latest_entry = max(((iter_num, completion) for iter_num, epoch_time, completion in entries if epoch_time <= target_time), default=(0, 0))
                    total_iter += latest_entry[0]
                    total_completion += latest_entry[1]
                    agent_count += 1

                total_iters.append(total_iter/max(agent_count, 1))
                completion_fractions.append(total_completion / max(agent_count * 100, 1))

            # Get latest values for legend
            base_scores = [s[2] for s in self.scores]
            extra_scores = [s[3] for s in self.scores]

            # Create compact plot with dual y-axis
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = ax1.twinx()

            # Plot scores on left axis with latest scores in labels
            ax1.plot(total_iters, base_scores, 'b-o',
                    label=f'Base ({base_scores[-1]:.3f} | {max(base_scores):.3f})',
                    linewidth=2, markersize=4)
            ax1.plot(total_iters, extra_scores, 'r-s',
                    label=f'Base+Extra ({extra_scores[-1]:.3f} | {max(extra_scores):.3f})',
                    linewidth=2, markersize=4)

            # Plot completion fraction on right axis with latest value in label
            ax2.plot(total_iters, completion_fractions, 'g--^',
                    label=f'Completion ({completion_fractions[-1]:.3f})',
                    linewidth=2, markersize=4, alpha=0.7)

            # Format axes
            ax1.set_xlabel('Average Agent Iterations')
            ax1.set_ylabel('Pass@1 Score', color='black')
            ax2.set_ylabel('Average Completion', color='green')
            ax1.set_title(f'{self.dataset.upper()} Scores vs Iterations ({len(agent_data)} agents)')

            ax1.set_ylim(0, 1)
            ax2.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

            plt.tight_layout()
            plt.savefig(self.plot_iteration, dpi=200, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Scores vs iterations plot failed: {e}")

    def setup_problems(self,
        num_problems: int = None,
        task_ids: List[str] = None) -> Dict:
        """Setup problems in benchmark directory"""
        # Get and filter problems
        problems = self.DATASETS[self.dataset]()
        if task_ids:
            problems = {tid: problems[tid] for tid in task_ids if tid in problems}
        if not num_problems: num_problems = 1e12
            # problems = dict(list(problems.items()))

        # Create problem directories
        self.problems = {}; self.skipped_problems = {}
        for i, (task_id, problem) in enumerate(problems.items()):
            if i < num_problems:
                safe_id = self._safe_task_id(task_id)
                problem_dir = self.benchmark_dir / safe_id
                problem_dir.mkdir(exist_ok=True)
                problem_file = problem_dir / f"{safe_id}.py"
                if not problem_file.exists():
                    problem_file.write_text(problem["prompt"])

                self.problems[task_id] = {"dir": problem_dir, "safe_id": safe_id}
            else:
                self.skipped_problems[task_id] = {}

        self.logger.info(f"Setup {len(self.problems)} {self.dataset} problems")
        return self.problems

    def evaluate(self,
        timeout: int = 1800,
        blocking: bool = True) -> Dict:
        """Run evaluation (blocking or non-blocking)"""
        try:
            solutions = self._extract_solutions()
            # Prepare evaluation
            solutions_file = self._prepare_eval(solutions)
            if not solutions_file:
                return {"error": "No valid solutions", "success": False}

            eval_script = self._create_script(solutions_file)

            if blocking:
                self.logger.info("Running evalplus.evaluate (blocking)")
                return self._run_blocking(eval_script, timeout)
            else:
                self.logger.info("Running evalplus.evaluate (non-blocking)")
                return {"success": self._run_non_blocking(eval_script)}
        except Exception as e:
            self.logger.error(f"Evaluate failed: {e}")
            self.logger.error(traceback.format_exc())
            return {"error": str(e), "success": False}

    def shutdown(self):
        """Terminate running evaluation"""
        if self._is_running():
            self.process.terminate()
            try:
                self.process.wait(timeout=5)  # Wait up to 5 seconds
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
        self.logger.info("Evaluator shutdown complete")

    # Private methods
    def _extract_solutions(self) -> List[Dict]:
        """Extract solutions from problem directories"""
        solutions = []
        for task_id, info in self.problems.items():
            solution_file = info["dir"] / f"{info['safe_id']}.py"
            solution = solution_file.read_text() if solution_file.exists() else ""
            solution_file = str(solution_file) if solution_file.exists() else ""
            solutions.append({"task_id": task_id, "solution": solution,
                "solution_file": solution_file})

        # Needed to avoid evalplus crashing from not enough problems
        for task_id, info in self.skipped_problems.items():
            solutions.append({"task_id": task_id, "solution": "", "solution_file": ""})

        return solutions

    def _is_running(self) -> bool:
        """Check if non-blocking evaluation is running"""
        return self.process and self.process.poll() is None

    def _safe_task_id(self, task_id: str) -> str:
        """Convert to filesystem-safe ID"""
        safe = task_id.replace("/", "_").replace("\\", "_")
        return f"task_{safe}" if safe and safe[0].isdigit() else safe

    def _prepare_eval(self, solutions) -> Optional[Path]:
        """Prepare evaluation directory and files"""
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        solutions_file = self.eval_dir / "solutions.jsonl"
        write_jsonl(str(solutions_file), solutions)

        # Clean old results
        for old_file in self.eval_dir.glob(f"*.eval_results.json"):
            old_file.unlink(missing_ok=True)

        return solutions_file

    def _create_script(self, solutions_file: Path) -> Path:
        """Create evaluation script with correct command ordering"""
        solutions_file = solutions_file.resolve()
        script = (self.eval_dir / "eval.sh").resolve()
        sanitized = Path(str(solutions_file).replace('.jsonl', '-sanitized.jsonl')).resolve()
        output_file = (self.eval_dir / f"eval.out.txt").resolve()

        content = f"""#!/bin/bash
CMD=$(command -v evalplus || echo "python -m evalplus")
EVAL_FILE=$([[ -f "{sanitized}" ]] && echo "{sanitized}" || echo "{solutions_file}")

echo "=== EVALPLUS {self.dataset}: $(wc -l < '{solutions_file}') solutions ===" | tee {output_file}
$CMD.sanitize --samples "{solutions_file}" 2>&1 || true
$CMD.evaluate --dataset {self.dataset} --samples "$EVAL_FILE" | tee -a {output_file}

echo "SOLUTIONS_FILE:{solutions_file}" | tee -a {output_file}
echo "EVALPLUS_RESULTS_FILE:${{EVAL_FILE%.jsonl}}.eval_results.json" | tee -a {output_file}
echo "EXIT_CODE:$?" | tee -a {output_file}
"""
        script.write_text(content)
        script.chmod(0o755)
        return script

    def _run_blocking(self, script: Path, timeout: int) -> Dict:
        """Run evaluation in blocking mode with score tracking"""
        timestamp = time.time()

        result = subprocess.run(
            ["bash", str(script)], capture_output=True, text=True,
            timeout=timeout, cwd=script.parent
        )
        output = f"{result.stdout}\n{result.stderr}\nEXIT_CODE:{result.returncode}"
        return_code = result.returncode; self.logger.info(output)

        # self.logger.debug(f"Eval script output:\n{output}")
        parsed_result = self._parse_result(output, return_code)

        # Record and plot scores with correct timestamp
        if parsed_result.get("success") and "scores" in parsed_result:
            # Use completion time, not start time
            completion_timestamp = time.time()
            self._save_and_plot_scores(parsed_result["scores"], completion_timestamp)

        return parsed_result

    def _run_non_blocking(self, script: Path) -> bool:
        """Start non-blocking evaluation with proper timestamp tracking"""
        if self._is_running():
            return False

        # Start process
        self.process = subprocess.Popen(
            ["bash", str(script)], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=script.parent
        )

        # Start monitoring thread for score tracking
        def monitor_process():
            stdout, stderr = self.process.communicate()
            # Ensure process is fully terminated
            self.process.wait()

            output = f"{stdout.decode()}\n{stderr.decode()}\nEXIT_CODE:{self.process.returncode}"
            # self.logger.debug(f"Eval script output:\n{output}")
            parsed_result = self._parse_result(output, self.process.returncode)

            # Record and plot scores with completion timestamp (not start timestamp)
            if parsed_result.get("success") and "scores" in parsed_result:
                completion_timestamp = time.time()  # ← FIX: Use completion time
                self._save_and_plot_scores(parsed_result["scores"], completion_timestamp)

            self.logger.info(f"Non-blocking evaluation completed: {parsed_result.get('success', False)}")

        monitor_thread = threading.Thread(target=monitor_process, daemon=True)
        monitor_thread.start()

        return True

    def _parse_result(self, output: str, exit_code: int) -> Dict:
        """Parse evaluation results"""
        if exit_code != 0:
            return {"error": "evaluation failed", "output": output, "success": False}

        self._create_eval_results(output)
        scores = self._parse_scores(output)
        result = {"output": output, "success": True}
        if scores:
            result["scores"] = scores
        return result

    def _create_eval_results(self, output: str):
        """Create JSON file mapping solution files to test results for passed problems"""
        passed_file = self.eval_dir / EVAL_RESULTS_NAME

        try:
            # Extract file paths from output
            solutions_file = None
            results_file = None

            for line in output.split('\n'):
                if line.startswith('SOLUTIONS_FILE:'):
                    solutions_file = Path(line.split(':', 1)[1].strip())
                elif line.startswith('EVALPLUS_RESULTS_FILE:'):
                    results_file = Path(line.split(':', 1)[1].strip())

            # Load solutions
            solutions = {}
            with open(solutions_file, 'r') as f:
                for line in f:
                    if line.strip():
                        sol = json.loads(line)
                        solutions[sol['task_id']] = sol

            # Load results
            with open(results_file, 'r') as f:
                results = json.load(f)

            # Create mapping for passed problems
            passed_mapping = {}
            for task_id, solution in solutions.items():
                if task_id in results['eval']:
                    test_results = results['eval'][task_id]

                    # Handle case where test_results is a list (take first entry)
                    if isinstance(test_results, list):
                        if not test_results: continue
                        test_result = test_results[0]  # Take first result
                    else:
                        test_result = test_results

                    solution_file_path = solution.get('solution_file')
                    if solution_file_path:
                        passed_mapping[solution_file_path] = {
                            'task_id': task_id,
                            'base_status': test_result.get('base_status'),
                            'plus_status': test_result.get('plus_status'),
                        }

            # Write passed solutions file
            with open(passed_file, 'w') as f:
                json.dump(passed_mapping, f, indent=4)

            self.logger.info(f"Created eval results file with {len(passed_mapping)} entries: {passed_file}")

        except Exception as e:
            self.logger.error(f"Failed to create eval results file: {passed_file}")
            self.logger.error(traceback.format_exc())

    def _parse_scores(self, output: str) -> Optional[Dict]:
        """Parse pass@k scores from output"""
        try:
            lines = output.strip().split('\n')
            scores = {}

            for i, line in enumerate(lines[:-1]):
                next_line = lines[i + 1].strip()
                if next_line.startswith("pass@") and '\t' in next_line:
                    key, value = next_line.split('\t', 1)
                    score = {key.rstrip(':'): float(value)}

                    if "base tests" in line.lower():
                        scores["base"] = score
                    elif "base + extra" in line.lower():
                        scores["base_plus_extra"] = score

            return scores or None
        except Exception:
            return None

    def get_scores_summary(self) -> Dict:
        """Get compact summary of score tracking"""
        if not self.scores:
            return {"total_evaluations": 0}

        # Get latest and first entries
        latest_entry = self.scores[-1]
        first_entry = self.scores[0]

        return {
            "total_evaluations": len(self.scores),
            "elapsed_minutes": latest_entry[1],
            "latest": {"base": latest_entry[2], "extra": latest_entry[3]},
            "improvement": {"base": latest_entry[2] - first_entry[2], "extra": latest_entry[3] - first_entry[3]},
            "plot_time_exists": self.plot_time.exists(),
            "plot_iteration_exists": self.plot_iteration.exists()
        }


class PeriodicEvaluator:
    """Enhanced PeriodicEvaluator with plotting support"""

    def __init__(self, evaluator, interval: int):
        self.evaluator = evaluator
        self.interval = interval
        self.logger = evaluator.logger
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_eval = 0

    def start_background(self):
        """Start periodic evaluation with immediate first run"""
        if self._thread and self._thread.is_alive():
            self.logger.error("PeriodicEvaluator already running")
            return

        self.logger.info(f"Starting PeriodicEvaluator with {self.interval}s ({self.interval/60:.1f} min) interval")
        self._stop_event.clear()
        self._last_eval = time.time() - self.interval

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        time.sleep(0.1)
        if not self._thread.is_alive():
            self.logger.error("Failed to start PeriodicEvaluator thread")

    def stop_background(self):
        """Stop periodic evaluation"""
        if not self._thread:
            return

        self._stop_event.set()

        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._thread = None
        self.logger.info("PeriodicEvaluator stopped")

    def _worker(self):
        """Worker that runs first evaluation immediately, then at intervals"""
        evaluation_count = 0

        while not self._stop_event.is_set():
            current_time = time.time()
            time_since_last = current_time - self._last_eval

            # Check if it's time to evaluate
            if time_since_last >= self.interval:
                evaluation_count += 1
                self.logger.info(f"Running background evaluation #{evaluation_count}")
                self._run_evaluation()
                self._last_eval = current_time

            # Wait for stop signal with short timeout to check timing frequently
            if self._stop_event.wait(timeout=min(60, self.interval/10)):
                break

    def _run_evaluation(self):
        """Run single evaluation with error handling"""
        result = self.evaluator.evaluate(blocking=False)

    def set_callback(self, set_callback_fn):
        """Set callback-based evaluation for agent integration"""
        def callback(agent, iteration):
            if time.time() - self._last_eval < self.interval:
                return

            if self.evaluator.evaluate(blocking=False).get("success"):
                self.logger.info(f"Background evaluation started (iteration {iteration})")
                self._last_eval = time.time()

        set_callback_fn(callback)
        # callback(None, 0) # Run immediately
