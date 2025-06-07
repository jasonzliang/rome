# evalplus_evaluator.py - Compact, modular EvalPlus evaluator
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rome.logger import get_logger

try:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
except ImportError:
    raise ImportError("EvalPlus not found. Install with: pip install evalplus")


class EvalplusEvaluator:
    """Compact evaluator for EvalPlus datasets with sync/async support"""

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

    # async def evaluate_async(self, timeout: int = 1800) -> Dict:
    #     """Run async evaluation"""

    #     try:
    #         self.logger.info("Running evalplus.evaluate (async version)")
    #         solutions = self._extract_solutions()
    #         solutions_file = self._prepare_eval(solutions)
    #         eval_script = self._create_script(solutions_file)

    #         process = await asyncio.create_subprocess_shell(
    #             f"bash {eval_script}",
    #             stdout=asyncio.subprocess.PIPE,
    #             stderr=asyncio.subprocess.PIPE,
    #             cwd=self.eval_dir  # FIXED: added self.
    #         )

    #         stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    #         output = f"{stdout.decode()}\n{stderr.decode()}\nEXIT_CODE:{process.returncode}"
    #         # self.logger.info(output)
    #         return self._parse_result(output, process.returncode)

    #     except asyncio.TimeoutError:
    #         if 'process' in locals():
    #             process.kill()
    #         self.logger.error("Async evaluate timed out")
    #         self.logger.error(traceback.format_exc())
    #         return {"error": "timeout", "success": False}
    #     except Exception as e:
    #         self.logger.error(f"Async evaluate failed: {e}")
    #         self.logger.error(traceback.format_exc())
    #         return {"error": str(e), "success": False}

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
        output_file = (self.eval_dir / f"eval.{int(time.time())}.out.txt").resolve()

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
        """Run evaluation in blocking mode"""
        result = subprocess.run(
            ["bash", str(script)], capture_output=True, text=True,
            timeout=timeout, cwd=script.parent
        )
        output = f"{result.stdout}\n{result.stderr}\nEXIT_CODE:{result.returncode}"
        self.logger.info(output)
        return self._parse_result(output, result.returncode)

    def _run_non_blocking(self, script: Path) -> bool:
        """Start non-blocking evaluation"""
        if self._is_running():
            return False
        self._process = subprocess.Popen(
            ["bash", str(script)], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, cwd=script.parent
        )
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


class PeriodicEvaluator:
    """Fixed PeriodicEvaluator with immediate first evaluation"""

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
        # self.logger.info("Worker started")
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
                # self.logger.info(f"Next evaluation in {self.interval}s ({self.interval/60:.1f} minutes)")

            # Wait for stop signal with short timeout to check timing frequently
            if self._stop_event.wait(timeout=min(60, self.interval/10)):  # Check every minute or 1/10 interval
                break
        # self.logger.info(f"Worker stopped after {evaluation_count} evaluations")

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
