import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ner import train as train_ner, process_file as run_ner
# NOTE: srl is imported lazily inside main() to avoid srl_init() running at
# module load time and potentially disabling torch autograd globally.
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

    btl1_clauses = os.path.join(base_dir, "..", "BTL1", "output", "clauses.txt")
    if not os.path.exists(btl1_clauses):
        sys.exit(1)

    ner_model_dir = os.path.join(model_dir, "ner_model")
    intent_model_dir = os.path.join(model_dir, "intent")
    ner_output = os.path.join(output_dir, "ner_results.json")
    srl_output = os.path.join(output_dir, "srl_results.json")
    intent_output = os.path.join(output_dir, "intent_classification.txt")
    if not args.inference_only:
        t = time.time()
        train_ner(data_dir, ner_model_dir, n_iter=30)
        t = time.time()
        train_intent(data_dir, intent_model_dir,
                     train_transformer=not args.no_transformer)
    t = time.time()
    run_ner(btl1_clauses, ner_output, ner_model_dir)
    t = time.time()
    from srl import process_file as run_srl  # Lazy import: keeps srl_init() away from training
    run_srl(btl1_clauses, srl_output, ner_output)
    t = time.time()
    run_intent(btl1_clauses, intent_output, intent_model_dir)


if __name__ == "__main__":
    main()
