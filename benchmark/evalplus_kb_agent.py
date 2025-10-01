#!/usr/bin/env python3
"""
Knowledge-Base Enhanced EvalPlus Evaluator with Progress Tracking

Dynamically creates specialized agents for each problem by:
1. Querying KB for required skills/knowledge/roles
2. Generating K agents with distinct personalities
3. Each agent queries KB for insights and generates solution with confidence
4. Merging solutions with confidence-weighted voting

Progress is tracked in a JSON file to enable resumption after interruptions.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rome.agent import Agent
from rome.logger import get_logger
from rome.config import load_config, merge_with_default_config
from rome.kb_client import ChromaClientManager
from benchmark.evalplus_evaluator import EvalplusEvaluator


class KnowledgeBaseEvaluator:
    """Knowledge-base enhanced evaluator with dynamic agent generation and progress tracking"""

    PROGRESS_FILE = "kb_eval_progress.json"

    def __init__(self, benchmark_dir: str, config_path: str,
                 dataset: str = "humaneval", num_agents: int = 3):
        self.benchmark_dir = Path(benchmark_dir)
        self.config_path = Path(config_path)
        self.dataset = dataset.lower()
        self.num_agents = num_agents
        self.progress_file = self.benchmark_dir / self.PROGRESS_FILE

        self.logger = get_logger()
        self.logger.configure({
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "console": True
        })

        self.config = self._load_config()
        self.evaluator = EvalplusEvaluator(self.benchmark_dir, self.dataset)
        self.coordinator = self._create_coordinator()

        kb_size = self.coordinator.kb_manager.size()
        if kb_size == 0:
            self.logger.error("Knowledge base is EMPTY - agents will operate without KB insights")
            self.logger.error("To populate KB, run standard benchmarks with save_insights=True")
        else:
            self.logger.info(f"Knowledge base contains {kb_size} documents")

        self.logger.info(f"KB Evalplus Evaluator initialized: {dataset}, {num_agents} agents per problem")

    def _load_config(self) -> Dict:
        """Load and merge configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        config = load_config(self.config_path)
        config = merge_with_default_config(config)
        config['Agent']['repository'] = str(self.benchmark_dir)
        config['Agent']['draw_fsm'] = False
        config['Agent']['save_insights'] = False
        config['Agent']['query_insights'] = True

        return config

    def _create_coordinator(self) -> Agent:
        """Create coordinator agent for KB operations"""
        return Agent(
            name='Coordinator',
            role='Expert coordinator for problem analysis and solution synthesis',
            config=self.config
        )

    def _load_progress(self) -> Dict:
        """Load progress from JSON file"""
        if not self.progress_file.exists():
            return {"completed": {}, "failed": {}}

        try:
            with open(self.progress_file) as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load progress file: {e}")
            return {"completed": {}, "failed": {}}

    def _save_progress(self, progress: Dict):
        """Save progress to JSON file"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")

    def _mark_completed(self, progress: Dict, task_id: str, result: Dict):
        """Mark problem as completed and save progress"""
        progress["completed"][task_id] = {
            "success": result.get("success"),
            "avg_confidence": result.get("avg_confidence"),
            "num_agents": result.get("num_agents")
        }
        self._save_progress(progress)

    def _mark_failed(self, progress: Dict, task_id: str, error: str):
        """Mark problem as failed and save progress"""
        progress["failed"][task_id] = {"error": error}
        self._save_progress(progress)

    def analyze_problem_requirements(self, problem: Dict) -> Dict:
        """Use KB to analyze what skills/knowledge/roles are needed"""
        kb_context = ""
        if self.coordinator.kb_manager.size() > 0:
            kb_response = self.coordinator.kb_manager.query(
                f"What skills and knowledge are needed to solve:\n{problem['prompt']}")
            if kb_response:
                kb_context = f"Relevant insights from knowledge base:\n{kb_response}"

        prompt = f"""Analyze this programming problem and identify:
1. Required technical skills (e.g., algorithms, data structures)
2. Domain knowledge needed (e.g., math, string processing)
3. Problem-solving approaches that would be effective

Problem:
{problem['prompt']}

{kb_context}

Respond with JSON:
{{
    "skills": ["skill1", "skill2", ...],
    "knowledge": ["domain1", "domain2", ...],
    "approaches": ["approach1", "approach2", ...]
}}
"""

        response = self.coordinator.chat_completion(
            prompt=prompt,
            response_format={"type": "json_object"}
        )

        return self.coordinator.parse_json_response(response) or {
            "skills": ["problem-solving"],
            "knowledge": ["programming"],
            "approaches": ["iterative"]
        }

    def generate_agent_personas(self, requirements: Dict, k: int) -> List[Dict]:
        """Generate K distinct agent personas based on requirements"""
        skills = ", ".join(requirements.get('skills', []))
        knowledge = ", ".join(requirements.get('knowledge', []))
        approaches = ", ".join(requirements.get('approaches', []))

        agent_prompt = f"""Create {k} distinct agent personas for solving a programming problem.
Each agent should have:
- A unique name
- A specific role/specialty
- A distinct problem-solving style

Required skills: {skills}
Domain knowledge: {knowledge}
Effective approaches: {approaches}

Respond with JSON object containing agents array:
{{
    "agents": [
        {{
            "name": "AgentName",
            "role": "Brief role description emphasizing skills and abilities",
            "style": "Problem-solving approach style"
        }},
        ...
    ]
}}

IMPORTANT:
 - Name must be between 8 and 32 characters long and alphanumeric only
 - Role should be 150-300 tokens long
 - Style should be 150-300 tokens long
"""

        response = self.coordinator.chat_completion(
            prompt=agent_prompt,
            response_format={"type": "json_object"}
        )

        result = self.coordinator.parse_json_response(response)
        personas = []
        if isinstance(result, dict):
            personas = result.get('agents', result.get('personas', []))
        elif isinstance(result, list):
            personas = result

        if not personas or len(personas) < k:
            self.logger.error(f"Generated only {len(personas)} personas, requested {k}, using fallback")
            personas = [
                {"name": f"Agent{i+1}", "role": f"Specialist {i+1}", "style": "systematic"}
                for i in range(k)
            ]

        return personas[:k]

    def solve_with_agent(self, persona: Dict, problem: Dict) -> Dict:
        """Have agent generate solution with confidence score"""
        temp_agent = Agent(
            name=persona['name'],
            role=persona['role'],
            config=self.config
        )

        kb_insights = ""
        if temp_agent.kb_manager.size() > 0:
            kb_query = f"How to solve:\n{problem['prompt']}"
            kb_response = temp_agent.kb_manager.query(kb_query)
            if kb_response:
                kb_insights = f"Relevant insights:\n{kb_response}"

        solve_prompt = f"""Your problem-solving style: {persona['style']}

{kb_insights}

Problem to solve:
{problem['prompt']}

Generate a complete Python solution and rate your confidence (0-100).

Respond with JSON:
{{
    "solution": "complete Python code",
    "confidence": [0-100],
    "reasoning": "brief explanation of approach"
}}

IMPORTANT:
- Solution must be this format: ```python your_code_here```
- Make sure to include all necessary import statements
- Solution must not contain any unnecessary code such as print statements or tests
"""

        response = temp_agent.chat_completion(
            prompt=solve_prompt,
            response_format={"type": "json_object"}
        )
        result = temp_agent.parse_json_response(response) or {}
        solution = temp_agent.parse_python_response(result.get('solution', ''))
        confidence = max(0, min(100, result.get('confidence', 50)))
        reasoning = result.get('reasoning', '')

        temp_agent.shutdown()
        del temp_agent

        return {
            "agent": persona['name'],
            "solution": solution,
            "confidence": confidence,
            "reasoning": reasoning
        }

    def merge_solutions(self, solutions: List[Dict], problem: Dict) -> str:
        """Merge solutions with confidence weighting"""
        if not solutions:
            return ""

        if len(solutions) == 1:
            return solutions[0]['solution']

        solutions_context = "\n\n".join([
            f"# Agent {s['agent']} (confidence: {s['confidence']}/100):\n"
            f"# Reasoning: {s['reasoning']}\n"
            f"# Solution:\n{s['solution']}"
            for s in solutions
        ])

        merge_prompt = f"""Synthesize the best solution from multiple agent proposals.
Give more weight to solutions with higher confidence scores.

## Original problem:
{problem['prompt']}

## Agent solutions:
{solutions_context}

## Your task
Create a single optimal solution that:
1. Incorporates the best ideas from high-confidence solutions
2. Resolves any conflicts or inconsistencies
3. Produces clean, working Python code

Respond with JSON:
{{
    "solution": "final merged Python code",
    "rationale": "brief explanation of merge decisions"
}}

IMPORTANT:
- Python code must be this format: ```python your_code_here```
- Make sure to include all necessary import statements
- Python code must not contain any unnecessary code such as print statements or tests
"""

        response = self.coordinator.chat_completion(
            prompt=merge_prompt,
            response_format={"type": "json_object"}
        )

        result = self.coordinator.parse_json_response(response) or {}
        merged = self.coordinator.parse_python_response(result.get('solution', ''))

        if not merged:
            solutions.sort(key=lambda x: x['confidence'], reverse=True)
            merged = solutions[0]['solution']

        return merged

    def solve_problem(self, task_id: str, problem: Dict) -> Dict:
        """Complete workflow for solving one problem"""
        try:
            self.logger.info(f"Solving {task_id}")

            requirements = self.analyze_problem_requirements(problem)
            self.logger.debug(f"Requirements: {requirements}")

            personas = self.generate_agent_personas(requirements, self.num_agents)
            self.logger.debug(f"Generated {len(personas)} personas")

            solutions = []
            for persona in personas:
                try:
                    solution = self.solve_with_agent(persona, problem)
                    solutions.append(solution)
                    self.logger.debug(f"{persona['name']}: confidence={solution['confidence']}")
                except Exception as e:
                    self.logger.error(f"Agent {persona['name']} failed: {e}")

            if not solutions:
                return {"success": False, "error": "No solutions generated"}

            final_solution = self.merge_solutions(solutions, problem)

            return {
                "success": True,
                "solution": final_solution,
                "num_agents": len(solutions),
                "avg_confidence": sum(s['confidence'] for s in solutions) / len(solutions),
                "requirements": requirements,
                "personas": personas
            }

        except Exception as e:
            self.logger.error(f"Problem {task_id} failed: {e}")
            self.logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

    def run_benchmark(self,
                     num_problems: Optional[int] = None,
                     task_ids: Optional[List[str]] = None,
                     run_evaluation: bool = True) -> Tuple[Dict, Path]:
        """Run complete KB-enhanced benchmark with progress tracking"""
        try:
            problems = self.evaluator.setup_problems(num_problems, task_ids)
            if not problems:
                raise ValueError("No problems setup")

            progress = self._load_progress()
            completed_count = len(progress["completed"])

            self.logger.info(f"Processing {len(problems)} problems ({completed_count} already completed)")

            results = {}
            dataset_problems = self.evaluator.DATASETS[self.dataset]()

            for task_id, info in problems.items():
                # Skip if already completed
                if task_id in progress["completed"]:
                    self.logger.info(f"⊘ {task_id}: Already completed, skipping")
                    results[task_id] = progress["completed"][task_id]
                    continue

                problem = dataset_problems[task_id]
                result = self.solve_problem(task_id, problem)
                results[task_id] = result

                if result.get('success'):
                    solution_file = info['dir'] / f"{info['safe_id']}.py"
                    solution_file.write_text(result['solution'])
                    self._mark_completed(progress, task_id, result)
                    self.logger.info(f"✓ {task_id}: {result['avg_confidence']:.1f}% avg confidence")
                else:
                    self._mark_failed(progress, task_id, result.get('error', 'Unknown error'))
                    self.logger.error(f"✗ {task_id}: {result.get('error')}")

            eval_results = {}
            if run_evaluation:
                self.logger.info("Running evalplus evaluation...")
                eval_results = self.evaluator.evaluate()

            results_file = self._save_results(results, eval_results)

            return {
                "problem_results": results,
                "evaluation_results": eval_results,
                "summary": self._create_summary(results, eval_results)
            }, results_file

        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            if self.coordinator:
                self.coordinator.shutdown()

    def _create_summary(self, results: Dict, eval_results: Dict) -> Dict:
        """Create summary statistics"""
        successful = sum(1 for r in results.values() if r.get('success'))
        total = len(results)

        avg_confidence = 0
        if successful > 0:
            confidences = [r['avg_confidence'] for r in results.values() if r.get('success')]
            avg_confidence = sum(confidences) / len(confidences)

        return {
            "total_problems": total,
            "successful_solutions": successful,
            "success_rate": f"{(successful/total*100):.1f}%",
            "avg_confidence": f"{avg_confidence:.1f}%",
            "agents_per_problem": self.num_agents,
            "scores": eval_results.get('scores')
        }

    def _save_results(self, results: Dict, eval_results: Dict) -> Path:
        """Save benchmark results"""
        output = {
            "benchmark_type": "kb_enhanced",
            "dataset": self.dataset,
            "num_agents_per_problem": self.num_agents,
            "problem_results": results,
            "evaluation_results": eval_results,
            "summary": self._create_summary(results, eval_results)
        }

        results_file = self.evaluator.log_dir / "kb_enhanced_results.json"
        results_file.write_text(json.dumps(output, indent=4, default=str))
        self.logger.info(f"Results saved: {results_file}")

        return results_file

    def print_summary(self, results: Dict):
        """Print benchmark summary"""
        summary = results.get('summary', {})

        self.logger.info("="*80)
        self.logger.info("KNOWLEDGE BASE EVALPLUS BENCHMARK SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Dataset: {self.dataset}")
        self.logger.info(f"Num Problems: {summary.get('total_problems')}")
        self.logger.info(f"Success Rate: {summary.get('success_rate')}")
        self.logger.info(f"Avg Confidence: {summary.get('avg_confidence')}")
        self.logger.info(f"Agents Per Prob: {summary.get('agents_per_problem')}")

        scores = summary.get('scores', {})
        if scores:
            self.logger.info("\nEvaluation Scores:")
            for score_type, metrics in scores.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        self.logger.info(
                            f"  {score_type.replace('_', ' ').title()} {metric}: {value}"
                        )
        self.logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Base EvalPlus Benchmark with Dynamic Agents"
    )
    parser.add_argument("benchmark_dir", help="Benchmark directory path")
    parser.add_argument("config_file", help="Agent configuration YAML")

    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--num-agents", type=int, default=1,
                       help="Number of agents per problem")
    parser.add_argument("--num-problems", type=int, help="Number of problems")
    parser.add_argument("--task-ids", nargs="+", help="Specific task IDs")
    parser.add_argument("--no-evaluation", action="store_true",
                       help="Skip final evaluation")

    args = parser.parse_args()

    evaluator = KnowledgeBaseEvaluator(
        args.benchmark_dir,
        args.config_file,
        args.dataset,
        args.num_agents
    )

    results, results_file = evaluator.run_benchmark(
        num_problems=args.num_problems,
        task_ids=args.task_ids,
        run_evaluation=not args.no_evaluation
    )

    evaluator.print_summary(results)
    evaluator.logger.info(f"Results: {results_file}")


if __name__ == "__main__":
    main()
