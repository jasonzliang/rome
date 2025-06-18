#!/usr/bin/env python3
"""
Script to analyze the relationship between test results and finished files.
Focuses on base tests only and shows:
1. Tests that passed with/without finished files
2. Tests that failed with/without finished files
3. Tests with finished files but no test results

Usage: python script.py result_directory_path
"""

import json
import argparse
import os
import sys
from pathlib import Path
from typing import Set, Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rome.config import LOG_DIR_NAME
from benchmark.evalplus_evaluator import EVAL_DIR_NAME

def extract_test_results(directory_path: str) -> Tuple[Set[str], Set[str]]:
    """Extract test IDs that passed and failed from eval_results.json"""
    directory = Path(directory_path)
    eval_results_path = directory / LOG_DIR_NAME / EVAL_DIR_NAME / "solutions-sanitized.eval_results.json"

    if not eval_results_path.exists():
        eval_results_path = directory / LOG_DIR_NAME / EVAL_DIR_NAME / "eval_results.json"

    if not eval_results_path.exists():
        print(f"❌ Error: eval_results.json not found in {directory / LOG_DIR_NAME / EVAL_DIR_NAME}/")
        sys.exit(1)

    with open(eval_results_path, 'r') as f:
        data = json.load(f)

    base_passed, base_failed = set(), set()
    eval_data = data.get('eval', {})

    for test_id, test_results in eval_data.items():
        if test_results:
            base_status = test_results[0].get('base_status')
            if base_status == 'pass':
                base_passed.add(test_id)
            elif base_status == 'fail':
                base_failed.add(test_id)

    return base_passed, base_failed

def extract_finished_tests(directory_path: str) -> Set[str]:
    """Extract test IDs that have finished files"""
    finished_tests = set()
    directory = Path(directory_path)

    for finished_file in directory.rglob("finished.json"):
        parts = finished_file.parts
        for part in parts:
            if part.startswith('HumanEval_'):
                test_num = part.split('_')[1]
                test_id = f"HumanEval/{test_num}"
                finished_tests.add(test_id)
                break

    return finished_tests

def analyze_test_coverage(passed: Set[str], failed: Set[str], finished: Set[str]) -> Dict[str, Set[str]]:
    """Compute test coverage analysis"""
    return {
        "passed_with_files": passed & finished,
        "passed_no_files": passed - finished,
        "failed_with_files": failed & finished,
        "failed_no_files": failed - finished,
        "files_no_results": finished - (passed | failed)
    }

def print_compact_list(items: Set[str], show_all: bool = False, max_items: int = 5) -> None:
    """Print a compact list of items"""
    if not items:
        return

    sorted_items = sorted(items)

    if show_all or len(sorted_items) <= max_items:
        print(f"    {', '.join(sorted_items)}")
    else:
        shown_items = sorted_items[:max_items]
        print(f"    {', '.join(shown_items)} ... (+{len(sorted_items) - max_items} more)")

def print_analysis_summary(analysis: Dict[str, Set[str]], show_all: bool = False) -> None:
    """Print comprehensive analysis summary"""
    total_tested = len(analysis["passed_with_files"]) + len(analysis["passed_no_files"]) + \
                   len(analysis["failed_with_files"]) + len(analysis["failed_no_files"])
    total_passed = len(analysis["passed_with_files"]) + len(analysis["passed_no_files"])
    total_failed = len(analysis["failed_with_files"]) + len(analysis["failed_no_files"])
    total_finished = len(analysis["passed_with_files"]) + len(analysis["failed_with_files"]) + \
                     len(analysis["files_no_results"])

    print(f"\n📊 TEST ANALYSIS SUMMARY")
    print(f"{'='*50}")

    # Core metrics
    print(f"Tests evaluated: {total_tested} (✅ {total_passed} passed, ❌ {total_failed} failed)")
    print(f"Finished files:  {total_finished}")

    # Coverage rates
    if total_passed > 0:
        passed_coverage = len(analysis["passed_with_files"]) / total_passed * 100
        print(f"Passed coverage: {passed_coverage:.1f}% have finished files")

    if total_failed > 0:
        failed_coverage = len(analysis["failed_with_files"]) / total_failed * 100
        print(f"Failed coverage: {failed_coverage:.1f}% have finished files")

    # Detailed breakdown
    print(f"\n📋 DETAILED BREAKDOWN")
    print(f"✅ Passed with finished files: {len(analysis['passed_with_files'])}")
    if analysis["passed_with_files"]:
        print_compact_list(analysis["passed_with_files"], show_all)

    print(f"❌ Failed with finished files: {len(analysis['failed_with_files'])}")
    if analysis["failed_with_files"]:
        print_compact_list(analysis["failed_with_files"], show_all)

    print(f"✅ Passed w/o finished files:  {len(analysis['passed_no_files'])}")
    if analysis["passed_no_files"]:
        print_compact_list(analysis["passed_no_files"], show_all)

    print(f"❌ Failed w/o finished files:  {len(analysis['failed_no_files'])}")
    if analysis["failed_no_files"]:
        print_compact_list(analysis["failed_no_files"], show_all)

    print(f"📁 Finished files w/o results: {len(analysis['files_no_results'])}")
    if analysis["files_no_results"]:
        print_compact_list(analysis["files_no_results"], show_all)

    # Attention items
    attention_items = []
    if analysis["passed_no_files"]:
        attention_items.append(f"⚠️  {len(analysis['passed_no_files'])} tests passed but missing finished files")
    if analysis["failed_with_files"]:
        attention_items.append(f"🔍 {len(analysis['failed_with_files'])} tests failed despite having finished files")
    if analysis["files_no_results"]:
        attention_items.append(f"📄 {len(analysis['files_no_results'])} finished files with no test results")

    if attention_items:
        print(f"\n🎯 ATTENTION ITEMS")
        for item in attention_items:
            print(f"   {item}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze relationship between test results and finished files"
    )
    parser.add_argument("directory_path", help="Path to result directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all items")
    args = parser.parse_args()

    if not Path(args.directory_path).exists():
        print(f"❌ Error: {args.directory_path} does not exist")
        sys.exit(1)

    try:
        print(f"🔍 Analyzing {args.directory_path}...")

        base_passed, base_failed = extract_test_results(args.directory_path)
        finished_tests = extract_finished_tests(args.directory_path)

        print(f"📊 Raw data: {len(base_passed)} passed, {len(base_failed)} failed, {len(finished_tests)} finished")

        analysis = analyze_test_coverage(base_passed, base_failed, finished_tests)
        print_analysis_summary(analysis, show_all=args.verbose)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
