#!/usr/bin/env python3
"""
Multi-LLM Judge for Answer Evaluation (Parallelized)
Evaluates LLM answers using GPT, Claude, and Gemini as judges.

Usage:
    python judge.py ./data --json             (Output JSON format)
    python judge.py ./data -t 3               (Run 3 trials per judge)
    python judge.py ./data -R --json -t 5     (Reasoning mode, JSON, 5 trials)
"""

import argparse
import concurrent.futures
import os
import threading
import traceback
import json
from pathlib import Path
from typing import Dict

import anthropic
import openai
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception

# --- Configuration ---

DEFAULT_RUBRIC_PATH = "config/llm_as_judge/prompts/nus_rubric_10pt_v2.txt"

RETRY_CONFIG = {
    "stop": stop_after_attempt(2),
    "wait": wait_exponential(multiplier=1, min=2, max=60),
}

PRINT_LOCK = threading.Lock()

def safe_print(*args, **kwargs):
    with PRINT_LOCK:
        print(*args, **kwargs)

JUDGES = {
    "gpt": {
        "model": "gpt-5.2",
        "client_init": lambda: openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), timeout=600.0),
    },
    "claude": {
        "model": "claude-opus-4-5-20251101",
        "client_init": lambda: anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=600.0),
    },
    "gemini": {
        "model": "gemini-3-pro-preview",
        "client_init": lambda: genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY"), http_options={'timeout': 600000}),
    },
}

# --- Formatting Templates ---

TEXT_FORMAT_INSTRUCTION = """
### Evaluation Output Instructions
- You MUST output your judgement in the following format. Output NOTHING else.
- Do not output markdown code blocks (```). Just the raw text.

#### Query: [Query text]

#### Score Card for [Agent File Name]
**Answer Summary:** [High level summary]

**1. New:**
* **Analysis:** [Justification]
* **Score:** [1-10]

**2. Useful:**
* **Analysis:** [Justification]
* **Score:** [1-10]

**3. Surprising:**
* **Analysis:** [Justification]
* **Score:** [1-10]

... [Repeat for all Agents] ...

#### Final Ranking
[Numbered list in descending order of total score]
"""

JSON_FORMAT_INSTRUCTION = """
### Evaluation Output Instructions
- You MUST output a valid JSON object.
- Do NOT output markdown formatting (like ```json ... ```). Output RAW JSON only.
- The JSON structure must be exactly:

{
  "query": "The original query text",
  "reviews": [
    {
      "agent_name": "answer_agent1.txt",
      "summary": "Short summary of answer...",
      "scores": {
        "New": 8,
        "Useful": 6,
        "Surprising": 9
      },
      "analysis": {
        "New": "Reasoning for New score...",
        "Useful": "Reasoning for Useful score...",
        "Surprising": "Reasoning for Surprising score..."
      }
    }
  ],
  "ranking": ["answer_agent1.txt", "answer_agent2.txt"]
}
"""

# --- Helper Functions ---

def should_retry_gemini(exception):
    if isinstance(exception, (ValueError, TypeError, AttributeError, KeyError)):
        return False
    return True

def load_rubric(rubric_path: Path) -> str:
    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric file not found: {rubric_path}")
    return rubric_path.read_text(encoding="utf-8")

def load_answers(directory: Path) -> Dict[str, str]:
    answers = {}
    for f in directory.glob("answer_*.txt"):
        if f.stat().st_size > 0:
            answers[f.name] = f.read_text(encoding="utf-8")
    return answers

def create_judge_prompt(query: str, answers: Dict[str, str], rubric: str, json_mode: bool) -> str:
    sep = "-" * 80
    prompt = f"#### QUERY FILE (query.txt):\n{query.strip()}\n\n{sep}\n\n"
    for filename, content in sorted(answers.items()):
        prompt += f"#### AGENT ANSWER FILE ({filename}):\n{content.strip()}\n\n"

    prompt += f"{sep}\n\n### SCORING RUBRIC\n{rubric.strip()}\n\n"

    # Dynamically append the correct formatting instruction
    if json_mode:
        prompt += JSON_FORMAT_INSTRUCTION
    else:
        prompt += TEXT_FORMAT_INSTRUCTION

    return prompt

# --- API Call Wrappers ---

@retry(**RETRY_CONFIG, retry=retry_if_exception_type((openai.RateLimitError, openai.APIError, openai.APITimeoutError)))
def call_gpt(client, prompt: str, use_reasoning: bool, json_mode: bool) -> str:
    kwargs = {
        "model": JUDGES["gpt"]["model"],
        "messages": [{"role": "user", "content": prompt}],
    }

    if json_mode and not use_reasoning:
        # Standard models support strict JSON mode
        kwargs["response_format"] = {"type": "json_object"}

    if use_reasoning:
        kwargs["reasoning_effort"] = "high"
    else:
        kwargs["temperature"] = 0.0
        kwargs["seed"] = 42
        kwargs["top_p"] = 1.0

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

@retry(**RETRY_CONFIG, retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIError, anthropic.APITimeoutError)))
def call_claude(client, prompt: str, use_reasoning: bool, json_mode: bool) -> str:
    kwargs = {
        "model": JUDGES["claude"]["model"],
        "max_tokens": 20000,
        "messages": [{"role": "user", "content": prompt}],
    }

    if use_reasoning:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}
        # Omit temperature for reasoning
    else:
        kwargs["temperature"] = 0.0

    response = client.messages.create(**kwargs)
    return next(block.text for block in response.content if block.type == "text")

@retry(**RETRY_CONFIG, retry=retry_if_exception(should_retry_gemini))
def call_gemini(client, prompt: str, use_reasoning: bool, json_mode: bool) -> str:
    gen_config_args = {}

    if json_mode:
        gen_config_args["response_mime_type"] = "application/json"

    if use_reasoning:
        gen_config_args["thinking_config"] = types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH)
    else:
        gen_config_args["temperature"] = 0.0
        gen_config_args["top_p"] = 1.0
        gen_config_args["top_k"] = 1

    response = client.models.generate_content(
        model=JUDGES["gemini"]["model"],
        contents=prompt,
        config=types.GenerateContentConfig(**gen_config_args),
    )
    return response.text

JUDGE_CALLS = {"gpt": call_gpt, "claude": call_claude, "gemini": call_gemini}

# --- Core Logic ---

def run_single_judge_task(directory: Path, judge_name: str, config: dict, prompt: str, output_file: Path, debug: bool, use_reasoning: bool, json_mode: bool):
    """
    Worker function executed by the thread pool.
    """
    try:
        mode_str = "Reasoning" if use_reasoning else "Standard"
        safe_print(f"⏳ Starting {judge_name} [Trial {output_file.name}]...")

        client = config["client_init"]()

        # Pass json_mode to the API call
        judgment = JUDGE_CALLS[judge_name](client, prompt, use_reasoning, json_mode)

        # Cleanup: Remove Markdown fences if the model added them despite instructions
        if json_mode:
            judgment = judgment.replace("```json", "").replace("```", "").strip()

        output_file.write_text(judgment, encoding="utf-8")
        safe_print(f"✅ Finished {output_file.name}")

        if debug:
            with PRINT_LOCK:
                print(f"DEBUG {output_file.name}: {judgment[:200]}...")

    except Exception as e:
        safe_print(f"❌ Error in {output_file.name}: {e}")
        if debug:
            traceback.print_exc()

def submit_directory_tasks(directory: Path, rubric: str, executor: concurrent.futures.ThreadPoolExecutor, args):
    """
    Submits tasks for all judges and all trials.
    """
    query_file = directory / "query.txt"
    if not query_file.exists():
        return

    answers = load_answers(directory)
    if not answers:
        return

    # Pass json_mode to prompt creation
    query = query_file.read_text(encoding="utf-8")
    prompt = create_judge_prompt(query, answers, rubric, args.json)

    # Loop for Judges
    for judge_name, config in sorted(JUDGES.items()):
        # Loop for Trials
        for i in range(1, args.trials + 1):

            # Naming convention: judge_gpt_1.json or judge_gpt_1.txt
            ext = "json" if args.json else "txt"
            suffix = f"_{i}" if args.trials > 1 else ""
            filename = f"judge_{judge_name}{suffix}.{ext}"

            output_file = directory / filename

            if not args.overwrite and output_file.exists() and output_file.stat().st_size > 0:
                safe_print(f"⏭️  Skipping {filename} (exists)")
                continue

            executor.submit(
                run_single_judge_task,
                directory,
                judge_name,
                config,
                prompt,
                output_file,
                args.debug,
                args.reasoning,
                args.json
            )

def find_and_judge_all(args, rubric: str):
    dirs_with_answers = [
        d for d in args.root_directory.rglob("*")
        if d.is_dir() and list(d.glob("answer_*.txt"))
    ]

    mode_msg = "🧠 REASONING" if args.reasoning else "🤖 DETERMINISTIC"
    format_msg = "JSON" if args.json else "TEXT"

    print(f"\nMode: {mode_msg} | Format: {format_msg} | Trials: {args.trials}")
    print(f"Found {len(dirs_with_answers)} directories.")
    print(f"Starting execution pool with {args.jobs} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for directory in sorted(dirs_with_answers):
            submit_directory_tasks(directory, rubric, executor, args)

    print("\n✨ All tasks completed.")

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-LLM Judge (Parallelized)")
    parser.add_argument("root_directory", type=Path)
    parser.add_argument("-r", "--rubric", type=Path, default=Path(DEFAULT_RUBRIC_PATH))
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-j", "--jobs", type=int, default=5)
    parser.add_argument("-R", "--reasoning", action="store_true", help="Enable reasoning/thinking")

    # New Arguments
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("-t", "--trials", type=int, default=1, help="Number of trials per judge (default: 1)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not args.root_directory.exists():
        exit(1)

    rubric = load_rubric(args.rubric)
    find_and_judge_all(args, rubric)