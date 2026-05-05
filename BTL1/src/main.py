"""
Assignment 1 - Main Pipeline
==============================
Preprocessing and Syntax Analysis

Runs the complete Assignment 1 pipeline or individual tasks:
  1.1  Clause Splitting      (input/raw_contracts.txt -> output/clauses.txt)
  1.2  Noun Phrase Chunking  (output/clauses.txt      -> output/chunks.txt)
  1.3  Dependency Analysis   (output/clauses.txt      -> output/dependency.json)

Usage:
    # Run all tasks
    python src/main.py

    # Run a specific task
    python src/main.py --task 1.1
    python src/main.py --task 1.2
    python src/main.py --task 1.3

    # Run multiple specific tasks
    python src/main.py --task 1.1 1.3
"""

import argparse
import os
import sys
import time

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clause_splitting import process_file as split_clauses
from noun_chunking import process_file as chunk_nouns
from dependency_analysis import process_file as analyze_dependencies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f"\n{title}")
    print("-" * 40)


def _check_input(path: str, label: str) -> None:
    """Exit with a clear message if a required input file is missing."""
    if not os.path.exists(path):
        print(f"\nERROR: Required input file not found: {path}")
        print(f"  '{label}' must be generated before this step.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Individual task runners
# ---------------------------------------------------------------------------

def run_task_1_1(raw_contracts: str, clauses_file: str) -> None:
    _check_input(raw_contracts, "input/raw_contracts.txt")
    _banner("[Task 1.1] Clause Splitting")
    start = time.time()
    split_clauses(raw_contracts, clauses_file)
    print(f"  Time: {time.time() - start:.2f}s")


def run_task_1_2(clauses_file: str, chunks_file: str) -> None:
    _check_input(clauses_file, "output/clauses.txt  (run Task 1.1 first)")
    _banner("[Task 1.2] Noun Phrase Chunking")
    start = time.time()
    chunk_nouns(clauses_file, chunks_file)
    print(f"  Time: {time.time() - start:.2f}s")


def run_task_1_3(clauses_file: str, dependency_file: str) -> None:
    _check_input(clauses_file, "output/clauses.txt  (run Task 1.1 first)")
    _banner("[Task 1.3] Dependency Analysis")
    start = time.time()
    analyze_dependencies(clauses_file, dependency_file)
    print(f"  Time: {time.time() - start:.2f}s")


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_MAP = {
    "1.1": run_task_1_1,
    "1.2": run_task_1_2,
    "1.3": run_task_1_3,
}

TASK_LABELS = {
    "1.1": "Clause Splitting",
    "1.2": "Noun Phrase Chunking",
    "1.3": "Dependency Analysis",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assignment 1 pipeline: Preprocessing and Syntax Analysis",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--task",
        nargs="+",
        choices=list(TASK_MAP.keys()),
        metavar="TASK",
        help=(
            "Task(s) to run. Choose one or more from: 1.1, 1.2, 1.3\n"
            "  1.1 — Clause Splitting\n"
            "  1.2 — Noun Phrase Chunking\n"
            "  1.3 — Dependency Analysis\n"
            "Omit to run all tasks in order."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # File paths
    raw_contracts   = os.path.join(input_dir, "raw_contracts.txt")
    clauses_file    = os.path.join(output_dir, "clauses.txt")
    chunks_file     = os.path.join(output_dir, "chunks.txt")
    dependency_file = os.path.join(output_dir, "dependency.json")

    # Resolve which tasks to run (preserve logical order even if user
    # supplies them out of order)
    ALL_TASKS = ["1.1", "1.2", "1.3"]
    tasks_to_run = sorted(set(args.task), key=ALL_TASKS.index) if args.task else ALL_TASKS

    # Header
    print("=" * 60)
    print("  Assignment 1: Preprocessing and Syntax Analysis")
    if args.task:
        labels = ", ".join(f"{t} ({TASK_LABELS[t]})" for t in tasks_to_run)
        print(f"  Running: {labels}")
    else:
        print("  Running: all tasks")
    print("=" * 60)

    # Dispatch
    runners = {
        "1.1": lambda: run_task_1_1(raw_contracts, clauses_file),
        "1.2": lambda: run_task_1_2(clauses_file, chunks_file),
        "1.3": lambda: run_task_1_3(clauses_file, dependency_file),
    }

    for task_id in tasks_to_run:
        runners[task_id]()

    # Summary
    output_map = {
        "1.1": clauses_file,
        "1.2": chunks_file,
        "1.3": dependency_file,
    }
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print("\n  Output files:")
    for task_id in tasks_to_run:
        print(f"    - {output_map[task_id]}")
    print()


if __name__ == "__main__":
    main()