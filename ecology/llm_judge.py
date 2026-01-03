#!/usr/bin/env python3
"""
Multi-LLM Judge for Answer Evaluation (Parallelized)
Evaluates LLM answers using GPT, Claude, and Gemini as judges.

Retry Strategy:
- All API calls retry up to 5 times on transient errors
- Exponential backoff: 2s, 4s, 8s, 16s, 32s (max 60s)
- GPT: Retries on rate limit, API errors, timeouts
- Claude: Retries on rate limit, API errors, timeouts
- Gemini: Retries on all errors except validation/programming errors

Dependencies:
    pip install anthropic openai google-genai tenacity
"""

import argparse
import concurrent.futures
import os
import threading
import traceback
from pathlib import Path
from typing import Dict

import anthropic
import openai
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception

# --- Configuration ---

# Default rubric path
DEFAULT_RUBRIC_PATH = "config/llm_as_judge/prompts/nus_rubric_10pt_v2.txt"

# Retry configuration
RETRY_CONFIG = {
    "stop": stop_after_attempt(2),
    "wait": wait_exponential(multiplier=1, min=2, max=60),
}

# Global Lock for thread-safe printing
PRINT_LOCK = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with PRINT_LOCK:
        print(*args, **kwargs)

# Judge configurations
JUDGES = {
    "gpt": {
        "model": "gpt-5.2",
        "client_init": lambda: openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=600.0),
    },
    "claude": {
        "model": "claude-sonnet-4-5-20250929",
        "client_init": lambda: anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=600.0),
    },
    "gemini": {
        "model": "gemini-3-pro-preview",
        "client_init": lambda: genai.Client(api_key=os.getenv("GOOGLE_API_KEY"), http_options={'timeout': 600000}),
    },
}

# --- Helper Functions ---

def should_retry_gemini(exception):
    """Determine if a Gemini exception should be retried."""
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

def create_judge_prompt(query: str, answers: Dict[str, str], rubric: str) -> str:
    prompt = f"{rubric}\n\n---\n\n#### Query File (query.txt):\n{query}\n\n"
    for filename, content in sorted(answers.items()):
        prompt += f"#### Answer File ({filename}):\n{content}\n\n"
    return prompt

# --- API Call Wrappers ---

@retry(**RETRY_CONFIG, retry=retry_if_exception_type((openai.RateLimitError, openai.APIError, openai.APITimeoutError)))
def call_gpt(client, prompt: str) -> str:
    response = client.chat.completions.create(
        model=JUDGES["gpt"]["model"],
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="high",
    )
    return response.choices[0].message.content

@retry(**RETRY_CONFIG, retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIError, anthropic.APITimeoutError)))
def call_claude(client, prompt: str) -> str:
    response = client.messages.create(
        model=JUDGES["claude"]["model"],
        max_tokens=20000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        messages=[{"role": "user", "content": prompt}],
    )
    return next(block.text for block in response.content if block.type == "text")

@retry(**RETRY_CONFIG, retry=retry_if_exception(should_retry_gemini))
def call_gemini(client, prompt: str) -> str:
    response = client.models.generate_content(
        model=JUDGES["gemini"]["model"],
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
        ),
    )
    return response.text

JUDGE_CALLS = {"gpt": call_gpt, "claude": call_claude, "gemini": call_gemini}

# --- Core Logic ---

def run_single_judge_task(directory: Path, judge_name: str, config: dict, prompt: str, output_file: Path, debug: bool):
    """
    Worker function executed by the thread pool.
    Runs a single judge for a single directory.
    """
    try:
        # Initial status print
        safe_print(f"⏳ Starting {judge_name} for {directory.name}...")

        # Initialize client inside the thread to ensure thread safety
        client = config["client_init"]()

        # Call the API
        judgment = JUDGE_CALLS[judge_name](client, prompt)

        # Write output
        output_file.write_text(judgment, encoding="utf-8")
        safe_print(f"✅ Finished {judge_name} for {directory.name}")

        if debug:
            with PRINT_LOCK:
                print(f"\n{'='*80}")
                print(f"DEBUG: {judge_name.upper()} Judgment ({directory.name}):")
                print("="*80)
                print(judgment[:1000] + "... [truncated]")
                print("="*80 + "\n")

    except Exception as e:
        safe_print(f"❌ Error in {judge_name} for {directory.name}: {e}")
        if debug:
            traceback.print_exc()

def submit_directory_tasks(directory: Path, rubric: str, executor: concurrent.futures.ThreadPoolExecutor, overwrite: bool, debug: bool):
    """
    Prepares the prompt for a directory and submits 3 judge tasks to the executor.
    """
    query_file = directory / "query.txt"
    if not query_file.exists():
        safe_print(f"⚠️  Skipping {directory.name}: No query.txt found")
        return

    answers = load_answers(directory)
    if not answers:
        safe_print(f"⚠️  Skipping {directory.name}: No answer files found")
        return

    # Build prompt once for all judges
    query = query_file.read_text(encoding="utf-8")
    prompt = create_judge_prompt(query, answers, rubric)

    if debug:
        with PRINT_LOCK:
            print(f"DEBUG: Generated prompt for {directory.name} ({len(prompt)} chars):")
            print(f"{prompt[:1000]}... [truncated]")

    # Submit a task for each judge
    for judge_name, config in sorted(JUDGES.items()):
        output_file = directory / f"judge_{judge_name}.txt"

        if not overwrite and output_file.exists() and output_file.stat().st_size > 0:
            safe_print(f"⏭️  Skipping {judge_name} for {directory.name} (exists)")
            continue

        executor.submit(
            run_single_judge_task,
            directory,
            judge_name,
            config,
            prompt,
            output_file,
            debug
        )

def find_and_judge_all(root_dir: Path, rubric: str, jobs: int, overwrite: bool = False, debug: bool = False):
    """Find all directories and process them using a thread pool."""
    dirs_with_answers = [
        d for d in root_dir.rglob("*")
        if d.is_dir() and list(d.glob("answer_*.txt"))
    ]

    print(f"Found {len(dirs_with_answers)} directories to process.")
    print(f"Starting execution pool with {jobs} workers...")

    # We use a context manager for the executor to ensure cleanup
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        # The context manager will wait for all tasks to complete before exiting this block
        for directory in sorted(dirs_with_answers):
            submit_directory_tasks(directory, rubric, executor, overwrite, debug)

    print("\n✨ All tasks completed.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-LLM Judge for Answer Evaluation (Parallelized)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("root_directory", type=Path, help="Root directory containing answer files")
    parser.add_argument("-r", "--rubric", type=Path, default=Path(DEFAULT_RUBRIC_PATH), help="Path to rubric file")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing judge files")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-j", "--jobs", type=int, default=5, help="Number of parallel workers (default: 5)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not args.root_directory.exists():
        print(f"Error: Directory '{args.root_directory}' does not exist")
        exit(1)

    try:
        rubric = load_rubric(args.rubric)
        print(f"Loaded rubric from: {args.rubric}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    find_and_judge_all(args.root_directory, rubric, args.jobs, args.overwrite, args.debug)
