import argparse
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clause_splitting import process_file as split_clauses
from noun_chunking import process_file as chunk_nouns
from dependency_analysis import process_file as analyze_dependencies
def _banner(title: str) -> None:
    pass


def _check_input(path: str, label: str) -> None:
    if not os.path.exists(path):
        sys.exit(1)


# ---------------------------------------------------------------------------
# Individual task runners
# ---------------------------------------------------------------------------

def run_task_1_1(raw_contracts: str, clauses_file: str) -> None:
    _check_input(raw_contracts, "input/raw_contracts.txt")
    split_clauses(raw_contracts, clauses_file)


def run_task_1_2(clauses_file: str, chunks_file: str) -> None:
    _check_input(clauses_file, "output/clauses.txt  (run Task 1.1 first)")
    chunk_nouns(clauses_file, chunks_file)


def run_task_1_3(clauses_file: str, dependency_file: str) -> None:
    _check_input(clauses_file, "output/clauses.txt  (run Task 1.1 first)")
    analyze_dependencies(clauses_file, dependency_file)


TASK_MAP = {
    "1.1": run_task_1_1,
    "1.2": run_task_1_2,
    "1.3": run_task_1_3,
}
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
        help=None,
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    raw_contracts   = os.path.join(input_dir, "raw_contracts.txt")
    clauses_file    = os.path.join(output_dir, "clauses.txt")
    chunks_file     = os.path.join(output_dir, "chunks.txt")
    dependency_file = os.path.join(output_dir, "dependency.json")
    ALL_TASKS = ["1.1", "1.2", "1.3"]
    tasks_to_run = sorted(set(args.task), key=ALL_TASKS.index) if args.task else ALL_TASKS
    runners = {
        "1.1": lambda: run_task_1_1(raw_contracts, clauses_file),
        "1.2": lambda: run_task_1_2(clauses_file, chunks_file),
        "1.3": lambda: run_task_1_3(clauses_file, dependency_file),
    }
    for task_id in tasks_to_run:
        runners[task_id]()

if __name__ == "__main__":
    main()