"""
Assignment 1 - Main Pipeline
==============================
Preprocessing and Syntax Analysis

Runs the complete Assignment 1 pipeline:
  1. Clause Splitting      (input/raw_contracts.txt -> output/clauses.txt)
  2. Noun Phrase Chunking   (output/clauses.txt     -> output/chunks.txt)
  3. Dependency Analysis    (output/clauses.txt     -> output/dependency.json)

Usage:
    python src/main.py
"""

import os
import sys
import time

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clause_splitting import process_file as split_clauses
from noun_chunking import process_file as chunk_nouns
from dependency_analysis import process_file as analyze_dependencies


def main():
    """Run the full Assignment 1 pipeline."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define file paths
    raw_contracts = os.path.join(input_dir, "raw_contracts.txt")
    clauses_file = os.path.join(output_dir, "clauses.txt")
    chunks_file = os.path.join(output_dir, "chunks.txt")
    dependency_file = os.path.join(output_dir, "dependency.json")

    # Check input file exists
    if not os.path.exists(raw_contracts):
        print(f"ERROR: Input file not found: {raw_contracts}")
        print("Please place your raw contract text in input/raw_contracts.txt")
        sys.exit(1)

    print("=" * 60)
    print("  Assignment 1: Preprocessing and Syntax Analysis")
    print("=" * 60)

    # ─── Task 1.1: Clause Splitting ──────────────────────────
    print("\n[Task 1.1] Clause Splitting")
    print("-" * 40)
    start = time.time()
    clauses = split_clauses(raw_contracts, clauses_file)
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s")

    # ─── Task 1.2: Noun Phrase Chunking ──────────────────────
    print("\n[Task 1.2] Noun Phrase Chunking")
    print("-" * 40)
    start = time.time()
    chunk_nouns(clauses_file, chunks_file)
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s")

    # ─── Task 1.3: Dependency Analysis ───────────────────────
    print("\n[Task 1.3] Dependency Analysis")
    print("-" * 40)
    start = time.time()
    analyze_dependencies(clauses_file, dependency_file)
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s")

    # ─── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"\n  Output files:")
    print(f"    - {clauses_file}")
    print(f"    - {chunks_file}")
    print(f"    - {dependency_file}")
    print()


if __name__ == "__main__":
    main()
