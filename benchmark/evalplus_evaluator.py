# evalplus_evaluator_enhanced.py - Enhanced with score tracking and plotting
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
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rome.logger import get_logger

try:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
except ImportError:
    raise ImportError("EvalPlus not found. Install with: pip install evalplus")


class EvalplusEvaluator:
    """Enhanced evaluator with score tracking and visualization"""

    DATASETS = {'humaneval': get_human_eval_plus, 'mbpp': get_mbpp_plus}

    def __init__(self,
        benchmark_dir: Union[str, Path],
        eval_dir: Union[str, Path],
        dataset: str = "humaneval"):

        if dataset not in self.DATASETS:
            raise ValueError(f"Dataset must be one of {list(self.DATASETS.keys())}")

        self.dataset = dataset
        self.benchmark_dir = Path(benchmark_dir)
        self.eval_dir = Path(eval_dir)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger()
        self.problems = {}
        self._process = None

        # Score tracking
        self._start_time = None
        self._scores_history = []
        self._scores_file = self.eval_dir / "scores_history.json"
        self._plot_file = self.eval_dir / "scores_plot.png"

        # Load existing history if available
        self._load_scores_history()

    def _load_scores_history(self):
        """Load existing scores history from file"""
        if self._scores_file.exists():
            try:
                with open(self._scores_file, 'r') as f:
                    data = json.load(f)
                    self._scores_history = data.get('scores_history', [])
                    self._start_time = data.get('start_time')
                self.logger.info(f"Loaded {len(self._scores_history)} historical score entries")
            except Exception as e:
                self.logger.error(f"Failed to load scores history: {e}")
                self._scores_history = []

    def _save_scores_history(self):
        """Save scores history to file"""
        try:
            data = {
                'start_time': self._start_time,
                'scores_history': self._scores_history,
                'dataset': self.dataset
            }
            with open(self._scores_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save scores history: {e}")

    def _record_scores(self, scores: Optional[Dict], timestamp: float = None):
        """Record scores with timestamp"""
        if not scores:
            return

        if self._start_time is None:
            self._start_time = timestamp or time.time()

        current_time = timestamp or time.time()
        elapsed_time = current_time - self._start_time

        # Extract pass@1 scores
        base_score = scores.get('base', {}).get('pass@1', 0.0)
        extra_score = scores.get('base_plus_extra', {}).get('pass@1', 0.0)

        entry = {
            'timestamp': current_time,
            'elapsed_time': elapsed_time,
            'base_score': base_score,
            'extra_score': extra_score,
            'datetime': datetime.fromtimestamp(current_time).isoformat()
        }

        self._scores_history.append(entry)
        self._save_scores_history()

        self.logger.info(f"Recorded scores at t={elapsed_time:.1f}s: base={base_score:.3f}, extra={extra_score:.3f}")

    def _create_scores_plot(self):
        """Create and save scores plot"""
        # if len(self._scores_history) < 2:
        #     self.logger.info("Insufficient data for plotting (need at least 2 points)")
        #     return

        try:
            # Extract data for plotting
            times = [entry['elapsed_time'] / 60 for entry in self._scores_history]  # Convert to minutes
            base_scores = [entry['base_score'] for entry in self._scores_history]
            extra_scores = [entry['extra_score'] for entry in self._scores_history]

            # Create plot
            plt.figure(figsize=(12, 8))

            # Plot lines
            plt.plot(times, base_scores, 'b-o', label='Base Tests', linewidth=2, markersize=6)
            plt.plot(times, extra_scores, 'r-s', label='Base + Extra Tests', linewidth=2, markersize=6)

            # Formatting
            plt.xlabel('Time Elapsed (minutes)', fontsize=12)
            plt.ylabel('Pass@1 Score', fontsize=12)
            plt.title(f'{self.dataset.upper()} Evaluation Scores Over Time', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)

            # Set y-axis to [0, 1] for pass rates
            plt.ylim(0, 1)

            # Add annotations for latest scores
            if times:
                latest_base = base_scores[-1]
                latest_extra = extra_scores[-1]
                latest_time = times[-1]

                plt.annotate(f'{latest_base:.3f}',
                           xy=(latest_time, latest_base),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.1),
                           fontsize=10)
                plt.annotate(f'{latest_extra:.3f}',
                           xy=(latest_time, latest_extra),
                           xytext=(10, -15), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.1),
                           fontsize=10)

            plt.tight_layout()
            plt.savefig(self._plot_file, dpi=150, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Updated scores plot: {self._plot_file}")

        except Exception as e:
            self.logger.error(f"Failed to create plot: {e}")
            self.logger.error(traceback.format_exc())

    def setup_problems(self,
        num_samples: int = None,
        task_ids: List[str] = None) -> Dict:
        """Setup problems in benchmark directory"""
        # Get and filter problems
        problems = self.DATASETS[self.dataset]()
        if task_ids:
            problems = {tid: problems[tid] for tid in task_ids if tid in problems}
        if num_samples:
            problems = dict(list(problems.items())[:num_samples])

        # Create problem directories
        self.problems = {}
        for task_id, problem in problems.items():
            safe_id = self._safe_task_id(task_id)
            problem_dir = self.benchmark_dir / safe_id
            problem_dir.mkdir(exist_ok=True)

            problem_file = problem_dir / f"{safe_id}.py"
            if not problem_file.exists():
                problem_file.write_text(problem["prompt"])

            self.problems[task_id] = {"dir": problem_dir, "safe_id": safe_id}

        self.logger.info(f"Setup {len(problems)} {self.dataset} problems")
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
            self._process.terminate()
            try:
                self._process.wait(timeout=5)  # Wait up to 5 seconds
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
        self.logger.info("Evaluator shutdown complete")

    # Private methods
    def _extract_solutions(self) -> List[Dict]:
        """Extract solutions from problem directories"""
        solutions = []
        for task_id, info in self.problems.items():
            solution_file = info["dir"] / f"{info['safe_id']}.py"
            solution = solution_file.read_text() if solution_file.exists() else ""
            solutions.append({"task_id": task_id, "solution": solution})

        return solutions

    def _is_running(self) -> bool:
        """Check if non-blocking evaluation is running"""
        return self._process and self._process.poll() is None

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
        for old_file in self.eval_dir.glob("*.eval_results.json"):
            old_file.unlink(missing_ok=True)

        return solutions_file

    def _create_script(self, solutions_file: Path) -> Path:
        """Create evaluation script"""
        solutions_file = solutions_file.resolve()
        script = (self.eval_dir / "eval.sh").resolve()
        sanitized = Path(str(solutions_file).replace('.jsonl', '-sanitized.jsonl')).resolve()
        output_file = (self.eval_dir / f"eval.out.txt").resolve()

        content = f"""#!/bin/bash
CMD=$(command -v evalplus || echo "python -m evalplus")
echo "=== EVALPLUS {self.dataset}: $(wc -l < '{solutions_file}') solutions ===" | tee {output_file}
$CMD.sanitize --samples "{solutions_file}" 2>&1 || true
EVAL_FILE=$([[ -f "{sanitized}" ]] && echo "{sanitized}" || echo "{solutions_file}")
$CMD.evaluate --dataset {self.dataset} --samples "$EVAL_FILE" | tee -a {output_file}
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
        self.logger.info(output)

        parsed_result = self._parse_result(output, result.returncode)

        # Record and plot scores
        if parsed_result.get("success") and "scores" in parsed_result:
            self._record_scores(parsed_result["scores"], timestamp)
            self._create_scores_plot()

        return parsed_result

    def _run_non_blocking(self, script: Path) -> bool:
        """Start non-blocking evaluation with score tracking"""
        if self._is_running():
            return False

        timestamp = time.time()

        # Start process
        self._process = subprocess.Popen(
            ["bash", str(script)], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=script.parent
        )

        # Start monitoring thread for score tracking
        def monitor_process():
            try:
                stdout, stderr = self._process.communicate()
                output = f"{stdout.decode()}\n{stderr.decode()}\nEXIT_CODE:{self._process.returncode}"

                parsed_result = self._parse_result(output, self._process.returncode)

                # Record and plot scores
                if parsed_result.get("success") and "scores" in parsed_result:
                    self._record_scores(parsed_result["scores"], timestamp)
                    self._create_scores_plot()

                self.logger.info(f"Non-blocking evaluation completed: {parsed_result.get('success', False)}")

            except Exception as e:
                self.logger.error(f"Error in non-blocking evaluation monitoring: {e}")

        monitor_thread = threading.Thread(target=monitor_process, daemon=True)
        monitor_thread.start()

        return True

    def _parse_result(self, output: str, exit_code: int) -> Dict:
        """Parse evaluation results"""
        if exit_code != 0:
            return {"error": "evaluation failed", "output": output, "success": False}

        scores = self._parse_scores(output)
        result = {"output": output, "success": True}
        if scores:
            result["scores"] = scores
        return result

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
        """Get summary of score tracking"""
        if not self._scores_history:
            return {"total_evaluations": 0}

        latest = self._scores_history[-1]
        first = self._scores_history[0]

        return {
            "total_evaluations": len(self._scores_history),
            "elapsed_time_minutes": latest['elapsed_time'] / 60,
            "latest_scores": {
                "base": latest['base_score'],
                "extra": latest['extra_score']
            },
            "improvement": {
                "base": latest['base_score'] - first['base_score'],
                "extra": latest['extra_score'] - first['extra_score']
            },
            "plot_file": str(self._plot_file) if self._plot_file.exists() else None
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
        try:
            result = self.evaluator.evaluate(blocking=False)
            if not result.get("success"):
                self.logger.error(f"Background evaluation failed: {result.get('error', 'unknown')}")
        except Exception as e:
            self.logger.error(f"Background evaluation error: {e}")

    def set_callback(self, set_callback_fn):
        """Set callback-based evaluation for agent integration"""
        def callback(agent, iteration):
            if time.time() - self._last_eval < self.interval:
                return

            if self.evaluator.evaluate(blocking=False).get("success"):
                self.logger.info(f"Background evaluation started (iteration {iteration})")
                self._last_eval = time.time()

        set_callback_fn(callback)
