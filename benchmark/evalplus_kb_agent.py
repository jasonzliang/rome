#!/usr/bin/env python3
"""
Knowledge-Base Enhanced EvalPlus Evaluator

Dynamically creates specialized agents for each problem by:
1. Querying KB for required skills/knowledge/roles
2. Generating K agents with distinct personalities
3. Each agent queries KB for insights and generates solution with confidence
4. Merging solutions with confidence-weighted voting

IMPORTANT: This script assumes the knowledge base is pre-populated with insights
from previous agent runs (using save_insights=True in config). The KB queries will
return empty results if the KB is unpopulated, in which case agents rely solely on
their base capabilities without additional context.

To populate the KB, run standard evalplus benchmarks with:
    Agent.save_insights = True
    Agent.use_ground_truth = True (optional, for higher quality insights)
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


class KBEnhancedEvaluator:
    """Knowledge-base enhanced evaluator with dynamic agent generation"""

    def __init__(self, benchmark_dir: str, config_path: str,
                 dataset: str = "humaneval", num_agents: int = 3):
        self.benchmark_dir = Path(benchmark_dir)
        self.config_path = Path(config_path)
        self.dataset = dataset.lower()
        self.num_agents = num_agents

        self.logger = get_logger()
        self.logger.configure({
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "console": True
        })

        # Load config and setup
        self.config = self._load_config()
        self.evaluator = EvalplusEvaluator(self.benchmark_dir, self.dataset)

        # Create a coordinator agent for KB access and orchestration
        self.coordinator = self._create_coordinator()

        # Check KB status and warn if empty
        kb_size = self.coordinator.kb_manager.size()
        if kb_size == 0:
            self.logger.warning("Knowledge base is EMPTY - agents will operate without KB insights")
            self.logger.warning("To populate KB, run standard benchmarks with save_insights=True")
        else:
            self.logger.info(f"Knowledge base contains {kb_size} documents")

        self.logger.info(f"KB-Enhanced Evaluator initialized: {dataset}, {num_agents} agents per problem")

    def _load_config(self) -> Dict:
        """Load and merge configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        config = load_config(str(self.config_path))
        config = merge_with_default_config(config)
        config['Agent']['repository'] = str(self.benchmark_dir)

        # Enable KB features
        config['Agent']['query_insights'] = True
        config['ChromaClientManager']['enable_reranking'] = True

        return config

    def _create_coordinator(self) -> Agent:
        """Create coordinator agent for KB operations"""
        coord_config = self.config.copy()
        coord_config['Agent']['name'] = 'Coordinator'
        coord_config['Agent']['role'] = 'Orchestrate problem analysis and solution synthesis'

        return Agent(
            name='Coordinator',
            role='Expert coordinator for problem analysis and solution synthesis',
            repository=str(self.benchmark_dir),
            config=coord_config
        )

    def analyze_problem_requirements(self, problem: Dict) -> Dict:
        """Use KB to analyze what skills/knowledge/roles are needed"""
        prompt = problem['prompt']

        query = f"""Analyze this programming problem and identify:
1. Required technical skills (e.g., algorithms, data structures)
2. Domain knowledge needed (e.g., math, string processing)
3. Problem-solving approaches that would be effective

Problem:
{prompt}

Respond with JSON:
{{
    "skills": ["skill1", "skill2", ...],
    "knowledge": ["domain1", "domain2", ...],
    "approaches": ["approach1", "approach2", ...]
}}"""

        # Query KB if available
        kb_context = ""
        if self.coordinator.kb_manager.size() > 0:
            kb_response = self.coordinator.kb_manager.query(
                f"What skills and knowledge are needed to solve: {prompt}")
            if kb_response:
                kb_context = f"\n\nRelevant insights from knowledge base:\n{kb_response}"

        response = self.coordinator.chat_completion(
            prompt=query + kb_context,
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

        prompt = f"""Create {k} distinct agent personas for solving a programming problem.
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
            "role": "Brief role description emphasizing unique perspective",
            "style": "Problem-solving approach style"
        }},
        ...
    ]
}}"""

        response = self.coordinator.chat_completion(
            prompt=prompt,
            response_format={"type": "json_object"}
        )

        result = self.coordinator.parse_json_response(response)

        # Extract personas from various possible structures
        personas = []
        if isinstance(result, dict):
            personas = result.get('agents', result.get('personas', []))
        elif isinstance(result, list):
            personas = result

        # Fallback if parsing fails or insufficient personas
        if not personas or len(personas) < k:
            self.logger.warning(f"Generated only {len(personas)} personas, requested {k}, using fallback")
            personas = [
                {"name": f"Agent{i+1}", "role": f"Specialist {i+1}", "style": "systematic"}
                for i in range(k)
            ]

        return personas[:k]

    def solve_with_agent(self, persona: Dict, problem: Dict) -> Dict:
        """Have agent generate solution with confidence score"""
        prompt_text = problem['prompt']

        # Query KB for relevant insights
        kb_insights = ""
        if self.coordinator.kb_manager.size() > 0:
            kb_query = f"How to solve: {prompt_text}"
            kb_response = self.coordinator.kb_manager.query(kb_query)
            if kb_response:
                kb_insights = f"\n\nRelevant insights:\n{kb_response}\n"

        # Generate solution
        solve_prompt = f"""You are {persona['name']}: {persona['role']}
Your problem-solving style: {persona['style']}

{kb_insights}
Problem to solve:
{prompt_text}

Generate a complete Python solution and rate your confidence (0-100).

For example, respond with JSON:
{{
    "solution": "complete Python code",
    "confidence": 50,
    "reasoning": "brief explanation of approach"
}}"""

        response = self.coordinator.chat_completion(
            prompt=solve_prompt,
            response_format={"type": "json_object"}
        )

        result = self.coordinator.parse_json_response(response) or {}

        return {
            "agent": persona['name'],
            "solution": result.get('solution', ''),
            "confidence": max(0, min(100, result.get('confidence', 50))),
            "reasoning": result.get('reasoning', '')
        }

    def merge_solutions(self, solutions: List[Dict], problem: Dict) -> str:
        """Merge solutions with confidence weighting"""
        if not solutions:
            return ""

        # If only one solution, return it
        if len(solutions) == 1:
            return solutions[0]['solution']

        # Build context of all solutions
        solutions_context = "\n\n".join([
            f"Agent {s['agent']} (confidence: {s['confidence']}/100):\n"
            f"Reasoning: {s['reasoning']}\n"
            f"Solution:\n{s['solution']}"
            for s in solutions
        ])

        merge_prompt = f"""Synthesize the best solution from multiple agent proposals.
Give more weight to solutions with higher confidence scores.

Original problem:
{problem['prompt']}

Agent solutions:
{solutions_context}

Create a single optimal solution that:
1. Incorporates the best ideas from high-confidence solutions
2. Resolves any conflicts or inconsistencies
3. Produces clean, working Python code

Respond with JSON:
{{
    "solution": "final merged Python code",
    "rationale": "brief explanation of merge decisions"
}}"""

        response = self.coordinator.chat_completion(
            prompt=merge_prompt,
            response_format={"type": "json_object"}
        )

        result = self.coordinator.parse_json_response(response) or {}
        merged = result.get('solution', '')

        # Fallback: use highest confidence solution
        if not merged:
            solutions.sort(key=lambda x: x['confidence'], reverse=True)
            merged = solutions[0]['solution']

        return merged

    def solve_problem(self, task_id: str, problem: Dict) -> Dict:
        """Complete workflow for solving one problem"""
        try:
            self.logger.info(f"Solving {task_id}")

            # Step 1: Analyze requirements
            requirements = self.analyze_problem_requirements(problem)
            self.logger.debug(f"Requirements: {requirements}")

            # Step 2: Generate agent personas
            personas = self.generate_agent_personas(requirements, self.num_agents)
            self.logger.debug(f"Generated {len(personas)} personas")

            # Step 3: Each agent generates solution
            solutions = []
            for persona in personas:
                try:
                    solution = self.solve_with_agent(persona, problem)
                    solutions.append(solution)
                    self.logger.debug(
                        f"{persona['name']}: confidence={solution['confidence']}"
                    )
                except Exception as e:
                    self.logger.error(f"Agent {persona['name']} failed: {e}")

            if not solutions:
                return {"success": False, "error": "No solutions generated"}

            # Step 4: Merge solutions
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
        """Run complete KB-enhanced benchmark"""
        try:
            # Setup problems
            problems = self.evaluator.setup_problems(num_problems, task_ids)
            if not problems:
                raise ValueError("No problems setup")

            self.logger.info(f"Processing {len(problems)} problems")

            # Solve each problem
            results = {}
            for task_id, info in problems.items():
                # Load problem from evaluator's dataset
                dataset_problems = self.evaluator.DATASETS[self.dataset]()
                problem = dataset_problems[task_id]

                # Solve using KB-enhanced approach
                result = self.solve_problem(task_id, problem)
                results[task_id] = result

                # Write solution to file
                if result.get('success'):
                    solution_file = info['dir'] / f"{info['safe_id']}.py"
                    solution_file.write_text(result['solution'])
                    self.logger.info(f"✓ {task_id}: {result['avg_confidence']:.1f}% avg confidence")
                else:
                    self.logger.warning(f"✗ {task_id}: {result.get('error')}")

            # Run evaluation if requested
            eval_results = {}
            if run_evaluation:
                self.logger.info("Running final evaluation...")
                eval_results = self.evaluator.evaluate()

            # Save results
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
        results_file.write_text(json.dumps(output, indent=2, default=str))
        self.logger.info(f"Results saved: {results_file}")

        return results_file

    def print_summary(self, results: Dict):
        """Print benchmark summary"""
        summary = results.get('summary', {})

        self.logger.info("="*80)
        self.logger.info("KB-ENHANCED EVALPLUS BENCHMARK SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Dataset: {self.dataset}")
        self.logger.info(f"Problems: {summary.get('total_problems')}")
        self.logger.info(f"Success Rate: {summary.get('success_rate')}")
        self.logger.info(f"Avg Confidence: {summary.get('avg_confidence')}")
        self.logger.info(f"Agents/Problem: {summary.get('agents_per_problem')}")

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
        description="KB-Enhanced EvalPlus Benchmark with Dynamic Agents"
    )
    parser.add_argument("benchmark_dir", help="Benchmark directory path")
    parser.add_argument("config_file", help="Agent configuration YAML")

    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--num-agents", type=int, default=3,
                       help="Number of agents per problem")
    parser.add_argument("--num-problems", type=int, help="Number of problems")
    parser.add_argument("--task-ids", nargs="+", help="Specific task IDs")
    parser.add_argument("--no-evaluation", action="store_true",
                       help="Skip final evaluation")

    args = parser.parse_args()

    evaluator = KBEnhancedEvaluator(
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
