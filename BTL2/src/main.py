"""
Assignment 2 - Main Pipeline
==============================
Information Extraction and Semantic Analysis

Phase 1 (Training):
  - Train spaCy NER model on annotated legal entities
  - Train TF-IDF + LogReg and DistilBERT for intent classification

Phase 2 (Inference):
  1. Custom NER              (clauses.txt -> output/ner_results.json)
  2. Semantic Role Labeling  (clauses.txt -> output/srl_results.json)
  3. Intent Classification   (clauses.txt -> output/intent_classification.txt)

Requires BTL1 output (clauses.txt) to be generated first.

Usage:
    python src/main.py                  # Train + Inference (full)
    python src/main.py --inference-only # Inference only (skip training)
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ner import train as train_ner, process_file as run_ner
from srl import process_file as run_srl
from intent_classification import train as train_intent, process_file as run_intent


def main():
    parser = argparse.ArgumentParser(description="Assignment 2 Pipeline")
    parser.add_argument("--inference-only", action="store_true",
                        help="Skip training, only run inference")
    parser.add_argument("--no-transformer", action="store_true",
                        help="Skip DistilBERT training (faster)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    model_dir = os.path.join(base_dir, "models")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Input: clauses from BTL1
    btl1_clauses = os.path.join(base_dir, "..", "BTL1", "output", "clauses.txt")
    if not os.path.exists(btl1_clauses):
        print("ERROR: BTL1 output not found. Run BTL1/src/main.py first.")
        sys.exit(1)

    # Paths
    ner_model_dir = os.path.join(model_dir, "ner_model")
    intent_model_dir = os.path.join(model_dir, "intent")
    ner_output = os.path.join(output_dir, "ner_results.json")
    srl_output = os.path.join(output_dir, "srl_results.json")
    intent_output = os.path.join(output_dir, "intent_classification.txt")

    print("=" * 60)
    print("  Assignment 2: Information Extraction & Semantic Analysis")
    print("=" * 60)

    # ═══ PHASE 1: TRAINING ═══════════════════════════════════
    if not args.inference_only:
        print("\n" + "=" * 60)
        print("  PHASE 1: MODEL TRAINING")
        print("=" * 60)

        # Train NER model
        print("\n[Training] NER Model (spaCy)")
        print("-" * 40)
        t = time.time()
        train_ner(data_dir, ner_model_dir, n_iter=30)
        print(f"  Time: {time.time() - t:.2f}s")

        # Train Intent Classification models
        print("\n[Training] Intent Classification")
        print("-" * 40)
        t = time.time()
        train_intent(data_dir, intent_model_dir,
                     train_transformer=not args.no_transformer)
        print(f"  Time: {time.time() - t:.2f}s")

    # ═══ PHASE 2: INFERENCE ══════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 2: INFERENCE")
    print("=" * 60)

    # Task 2.1: NER
    print("\n[Task 2.1] Named Entity Recognition")
    print("-" * 40)
    t = time.time()
    run_ner(btl1_clauses, ner_output, ner_model_dir)
    print(f"  Time: {time.time() - t:.2f}s")

    # Task 2.2: SRL
    print("\n[Task 2.2] Semantic Role Labeling")
    print("-" * 40)
    t = time.time()
    run_srl(btl1_clauses, srl_output, ner_output)
    print(f"  Time: {time.time() - t:.2f}s")

    # Task 2.3: Intent Classification
    print("\n[Task 2.3] Clause Intent Classification")
    print("-" * 40)
    t = time.time()
    run_intent(btl1_clauses, intent_output, intent_model_dir)
    print(f"  Time: {time.time() - t:.2f}s")

    # ═══ SUMMARY ═════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"\n  Models saved to:")
    print(f"    - {ner_model_dir}")
    print(f"    - {intent_model_dir}")
    print(f"\n  Output files:")
    print(f"    - {ner_output}")
    print(f"    - {srl_output}")
    print(f"    - {intent_output}")
    print()


if __name__ == "__main__":
    main()
