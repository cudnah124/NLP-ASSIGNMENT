"""
Task 2.1: Custom Named Entity Recognition (NER)
==================================================
Train a spaCy NER model on domain-specific legal entities, then use it
for inference on contract clauses.

Entity schema (6 classes):
  PARTY    — Contracting parties (e.g., Party A, Party B, Employer, Employee)
  MONEY    — Financial quantities (e.g., 10,000,000 VND, USD 500)
  DATE     — Dates and deadlines (e.g., 01/01/2024, within 30 days)
  RATE     — Percentage rates (e.g., 1% per day, 5% interest rate)
  PENALTY  — Penalty clauses (e.g., a penalty of …, liquidated damages)
  LAW      — Legal references (e.g., Civil Code 2015, Article 45)

Pipeline:
  1. Load annotated training data from data/ner_training_data.json
  2. Validate entity labels against the official schema
  3. Train a spaCy NER model (blank English model)
  4. Evaluate per epoch and per label; save training plot
  5. Save the trained model to models/ner_model/
  6. Run inference on clauses → output/ner_results.json

Input:  output/clauses.txt              (from Assignment 1)
        data/ner_training_data.json     (annotated by students)
Output: output/ner_results.json

Training data format (ner_training_data.json):
[
  {
    "text": "Party A shall pay 10,000,000 VND by 31/12/2024.",
    "entities": [
      [0,  7,  "PARTY"],
      [18, 34, "MONEY"],
      [38, 48, "DATE"]
    ]
  },
  ...
]

Output format (ner_results.json):
[
  {
    "clause": "Party A shall pay 10,000,000 VND by 31/12/2024.",
    "entities": [
      {"text": "Party A",        "label": "PARTY",   "start_char": 0,  "end_char": 7},
      {"text": "10,000,000 VND", "label": "MONEY",   "start_char": 18, "end_char": 34},
      {"text": "31/12/2024",     "label": "DATE",    "start_char": 38, "end_char": 48}
    ]
  },
  ...
]
"""

import json
import os
import random
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding


# ─── Official entity schema ───────────────────────────────────

ENTITY_LABELS = {"PARTY", "MONEY", "DATE", "RATE", "PENALTY", "LAW"}


# ─── Training data ───────────────────────────────────────────

def load_training_data(data_path):
    """
    Load and validate annotated NER training data from JSON.

    Each record must have:
      "text"     : str
      "entities" : [[start, end, "LABEL"], ...]

    Labels are validated against ENTITY_LABELS. Unknown labels are
    skipped with a warning; overlapping spans within the same record
    are also removed to prevent spaCy alignment errors.

    Returns:
        List of (text, {"entities": [(start, end, label), ...]}) tuples.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    training_data = []
    skipped_labels = set()

    for item in raw_data:
        text = item["text"]
        raw_entities = [tuple(e) for e in item["entities"]]

        # Validate labels
        valid_entities = []
        for start, end, label in raw_entities:
            if label not in ENTITY_LABELS:
                skipped_labels.add(label)
                continue
            valid_entities.append((start, end, label))

        # Remove overlapping spans (keep the first one by start position)
        valid_entities = _remove_overlaps(valid_entities)

        if valid_entities:
            training_data.append((text, {"entities": valid_entities}))

    if skipped_labels:
        print(f"  [Warning] Skipped unknown labels: {sorted(skipped_labels)}")
        print(f"  [Info]    Valid schema: {sorted(ENTITY_LABELS)}")

    return training_data


def _remove_overlaps(entities):
    """
    Remove overlapping entity spans, keeping the earlier-starting span.
    Entities are sorted by start offset; any span that overlaps with the
    last accepted span is discarded.
    """
    sorted_ents = sorted(entities, key=lambda e: (e[0], e[1]))
    result = []
    last_end = -1
    for start, end, label in sorted_ents:
        if start >= last_end:
            result.append((start, end, label))
            last_end = end
    return result


# ─── Training ────────────────────────────────────────────────

def train_ner_model(training_data, output_dir, n_iter=30):
    """
    Train a blank spaCy English NER model on the provided training data.

    Evaluation is performed every 5 epochs (and at epoch 1) on a held-out
    20% split. Per-label P/R/F1 is printed alongside overall metrics.

    Args:
        training_data: List of (text, {"entities": [...]}) tuples.
        output_dir:    Path to save the trained model.
        n_iter:        Number of training iterations (default: 30).

    Returns:
        The trained spaCy nlp model.
    """
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner", last=True)

    # Register all labels found in training data (must be in ENTITY_LABELS)
    labels = set()
    for _, annotations in training_data:
        for ent in annotations["entities"]:
            labels.add(ent[2])
    for label in sorted(labels):
        ner.add_label(label)

    print(f"  Entity labels: {sorted(labels)}")

    # Train / eval split (80 / 20)
    random.shuffle(training_data)
    split_idx = int(len(training_data) * 0.8)
    train_data = training_data[:split_idx]
    eval_data  = training_data[split_idx:]

    print(f"  Train samples : {len(train_data)}")
    print(f"  Eval  samples : {len(eval_data)}")
    print(f"  Iterations    : {n_iter}")
    print()

    optimizer = nlp.begin_training()

    # History for plot
    epoch_hist = []
    loss_hist  = []
    f1_hist    = []

    for epoch in range(n_iter):
        random.shuffle(train_data)
        losses = {}

        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            examples = [
                Example.from_dict(nlp.make_doc(text), ann)
                for text, ann in batch
            ]
            nlp.update(examples, sgd=optimizer, drop=0.35, losses=losses)

        # Evaluate periodically
        if (epoch + 1) % 5 == 0 or epoch == 0:
            eval_examples = [
                Example.from_dict(nlp.make_doc(text), ann)
                for text, ann in eval_data
            ]
            scores = nlp.evaluate(eval_examples)

            overall_p  = scores.get("ents_p", 0.0)
            overall_r  = scores.get("ents_r", 0.0)
            overall_f1 = scores.get("ents_f", 0.0)
            per_label  = scores.get("ents_per_type", {})

            print(f"  Epoch {epoch + 1:2d}/{n_iter}"
                  f" | Loss: {losses.get('ner', 0):7.2f}"
                  f" | P: {overall_p:.3f}"
                  f" | R: {overall_r:.3f}"
                  f" | F1: {overall_f1:.3f}")

            # Per-label breakdown
            for lbl in sorted(per_label):
                lp = per_label[lbl].get("p", 0.0)
                lr = per_label[lbl].get("r", 0.0)
                lf = per_label[lbl].get("f", 0.0)
                print(f"           {lbl:<10s}"
                      f" P: {lp:.3f} | R: {lr:.3f} | F1: {lf:.3f}")
            print()

            epoch_hist.append(epoch + 1)
            loss_hist.append(losses.get("ner", 0))
            f1_hist.append(overall_f1)

    _save_training_plot(epoch_hist, loss_hist, f1_hist, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"  Model saved to: {output_dir}")

    return nlp


def _save_training_plot(epoch_hist, loss_hist, f1_hist, model_output_dir):
    """Save a dual-axis Loss / F1 training curve to report_assets/."""
    try:
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Loss", color="tab:red")
        ax1.plot(epoch_hist, loss_hist, color="tab:red", marker="o", label="Loss")
        ax1.tick_params(axis="y", labelcolor="tab:red")
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Eval F1-Score", color="tab:blue")
        ax2.plot(epoch_hist, f1_hist, color="tab:blue", marker="s", label="F1")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax2.set_ylim(0, 1)

        plt.title("NER Training: Loss vs F1-Score")
        fig.tight_layout()

        assets_dir = os.path.join(
            os.path.dirname(os.path.dirname(model_output_dir)), "report_assets"
        )
        os.makedirs(assets_dir, exist_ok=True)
        plot_path = os.path.join(assets_dir, "ner_training_history.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"  Training plot saved: {plot_path}")
    except ImportError:
        print("  [Info] matplotlib not installed — training plot skipped.")


def load_trained_model(model_dir):
    """Load a previously saved spaCy NER model from disk."""
    return spacy.load(model_dir)


# ─── Inference ───────────────────────────────────────────────

def recognize_entities(clause_text, nlp):
    """
    Run NER inference on a single clause.

    Args:
        clause_text: A single clause string.
        nlp:         A trained spaCy NER model.

    Returns:
        {
          "clause":   str,
          "entities": [
            {"text": str, "label": str, "start_char": int, "end_char": int},
            ...
          ]
        }
    """
    doc = nlp(clause_text.strip())
    entities = [
        {
            "text":       ent.text,
            "label":      ent.label_,
            "start_char": ent.start_char,
            "end_char":   ent.end_char,
        }
        for ent in doc.ents
    ]
    return {"clause": clause_text.strip(), "entities": entities}


# ─── Public pipeline entry points ────────────────────────────

def train(data_dir, model_dir, n_iter=30):
    """
    Full training pipeline: load data → train → save model.

    Args:
        data_dir:  Directory containing ner_training_data.json.
        model_dir: Directory to save the trained model.
        n_iter:    Number of training iterations.

    Returns:
        Trained spaCy nlp model.
    """
    data_path = os.path.join(data_dir, "ner_training_data.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    print(f"  Loading training data from: {data_path}")
    training_data = load_training_data(data_path)
    print(f"  Loaded {len(training_data)} valid training records.")
    return train_ner_model(training_data, model_dir, n_iter=n_iter)


def process_file(input_path, output_path, model_dir=None):
    """
    Run NER inference on all clauses in a file.

    Args:
        input_path:  Path to clauses.txt (one clause per line).
        output_path: Path to output ner_results.json.
        model_dir:   Path to the trained model directory.
    """
    if not (model_dir and os.path.exists(model_dir)):
        raise FileNotFoundError(
            f"Trained NER model not found at: {model_dir}\n"
            "  Please run training first (--mode train)."
        )

    nlp = load_trained_model(model_dir)
    print(f"  Loaded model from: {model_dir}")

    with open(input_path, "r", encoding="utf-8") as f:
        clauses = [line.strip() for line in f if line.strip()]

    results = []
    label_counts = {lbl: 0 for lbl in ENTITY_LABELS}
    total_entities = 0

    for clause in clauses:
        result = recognize_entities(clause, nlp)
        results.append(result)
        for ent in result["entities"]:
            total_entities += 1
            lbl = ent["label"]
            if lbl in label_counts:
                label_counts[lbl] += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Clauses processed : {len(clauses)}")
    print(f"  Total entities    : {total_entities}")
    print("  Entities by label :")
    for lbl in sorted(ENTITY_LABELS):
        print(f"    {lbl:<10s}: {label_counts[lbl]}")
    print(f"  Output            : {output_path}")

    return results


# ─── CLI ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Task 2.1: Custom NER — train or run inference",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "all"],
        default="all",
        help=(
            "train — train the NER model only\n"
            "infer — run inference only (model must exist)\n"
            "all   — train then infer (default)"
        ),
    )
    parser.add_argument("--iter", type=int, default=30,
                        help="Number of training iterations (default: 30)")
    args = parser.parse_args()

    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir   = os.path.join(base_dir, "data")
    model_dir  = os.path.join(base_dir, "models", "ner_model")
    clauses_in = os.path.join(base_dir, "input", "clauses.txt")
    output_path = os.path.join(base_dir, "output", "ner_results.json")

    if args.mode in ("train", "all"):
        print("=" * 50)
        print("  [Task 2.1] Training NER Model")
        print("=" * 50)
        train(data_dir, model_dir, n_iter=args.iter)

    if args.mode in ("infer", "all"):
        print("=" * 50)
        print("  [Task 2.1] Running NER Inference")
        print("=" * 50)
        process_file(clauses_in, output_path, model_dir)