# evalplus_evaluator.py - Compact, modular EvalPlus evaluator
import asyncio
import json
import re
import os
import signal
import sys
import subprocess
import traceback
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
                return self._run_blocking(eval_script, timeout)
            else:
                return {"success": self._run_non_blocking(eval_script)}
        except Exception as e:
            self.logger.error(f"Evaluate failed: {e}")
            self.logger.error(traceback.format_exc())
            return {"error": str(e), "success": False}

    async def evaluate_async(self, timeout: int = 1800) -> Dict:
        """Run async evaluation"""

        try:
            solutions = self._extract_solutions()
            solutions_file = self._prepare_eval(solutions)
            eval_script = self._create_script(solutions_file)

            process = await asyncio.create_subprocess_shell(
                f"bash {eval_script}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.eval_dir  # FIXED: added self.
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
            output = f"{stdout.decode()}\n{stderr.decode()}\nEXIT_CODE:{process.returncode}"
            return self._parse_result(output, process.returncode)

        except asyncio.TimeoutError:
            if 'process' in locals():
                process.kill()
            self.logger.error("Async evaluate timed out")
            self.logger.error(traceback.format_exc())
            return {"error": "timeout", "success": False}
        except Exception as e:
            self.logger.error(f"Async evaluate failed: {e}")
            self.logger.error(traceback.format_exc())
            return {"error": str(e), "success": False}

    def shutdown(self):
        """Terminate running evaluation"""
        if self._is_running():
            self._process.terminate()
            try:
                self._process.wait(timeout=5)  # Wait up to 10 seconds
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
        output_file = (self.eval_dir / "eval.output.txt").resolve()

        content = f"""#!/bin/bash
CMD=$(command -v evalplus || echo "python -m evalplus")
echo "=== EVALPLUS {self.dataset}: $(wc -l < '{solutions_file}') solutions ===" | tee {output_file}
$CMD.sanitize --samples "{solutions_file}" 2>&1 | tee -a {output_file} || true
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
    """Periodic evaluation wrapper"""

    def __init__(self, evaluator: EvalplusEvaluator, interval: int):
        self.evaluator = evaluator
        self.interval = interval

        self.logger = get_logger()
        self.last_eval_time = 0
        self._task = None
        self._stop = asyncio.Event()

    async def start_async(self):
        """Start async periodic evaluation"""
        if self._task:
            return

        self._stop.clear()
        self._task = asyncio.create_task(self._eval_worker())

    async def _eval_worker(self):
        """Async evaluation worker"""
        while not self._stop.is_set():
            try:
                await asyncio.sleep(self.interval)
                if time.time() - self.last_eval_time >= self.interval:
                    await self.evaluator.evaluate_async()
                    self.last_eval_time = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Periodic evaluation failed: {e}")
                self.logger.error(traceback.format_exc())

    async def stop_async(self):
            """Stop periodic evaluation"""
            if self._task:
                self._stop.set()
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                self._task = None

    def set_callback(self, set_callback_fn: Callable):
        """Start callback-based periodic evaluation"""
        def callback(agent, iteration):
            if time.time() - self.last_eval_time < self.interval:
                return
            if self.evaluator.evaluate(blocking=False)["success"]:
                self.logger.info(f"Background eval started (iteration {iteration})")
                self.last_eval_time = time.time()

        set_callback_fn(callback)
