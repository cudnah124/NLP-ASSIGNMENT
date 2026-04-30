"""
Task 2.1: Custom Named Entity Recognition (NER)
==================================================
Train a spaCy NER model on domain-specific legal entities, then use it
for inference on contract clauses.

Entity schema: PARTY, MONEY, DATE, RATE, DURATION, LAW

Pipeline:
  1. Load annotated training data from data/ner_training_data.json
  2. Train a spaCy NER model (blank or fine-tuned)
  3. Save the trained model to models/ner_model/
  4. Run inference on clauses and write output to output/ner_results.json

Input:  Clauses from Assignment 1 (output/clauses.txt)
Output: output/ner_results.json
"""

import json
import os
import random
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding


# ─── Training ────────────────────────────────────────────────

def load_training_data(data_path):
    """
    Load annotated NER training data from JSON.

    Expected format:
    [
      {"text": "...", "entities": [[start, end, "LABEL"], ...]},
      ...
    ]

    Returns data in spaCy training format:
    [("text", {"entities": [(start, end, "LABEL"), ...]}), ...]
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    training_data = []
    for item in raw_data:
        text = item["text"]
        entities = [tuple(e) for e in item["entities"]]
        training_data.append((text, {"entities": entities}))

    return training_data


def train_ner_model(training_data, output_dir, n_iter=30):
    """
    Train a spaCy NER model from scratch using annotated data.

    Args:
        training_data: List of (text, {"entities": [...]}) tuples.
        output_dir:    Path to save the trained model.
        n_iter:        Number of training iterations.

    Returns:
        The trained spaCy model (nlp).
    """
    # Create a blank English model
    nlp = spacy.blank("en")

    # Add the NER pipeline component
    ner = nlp.add_pipe("ner", last=True)

    # Add custom entity labels
    labels = set()
    for _, annotations in training_data:
        for ent in annotations["entities"]:
            labels.add(ent[2])

    for label in labels:
        ner.add_label(label)

    print(f"  Entity labels: {sorted(labels)}")
    
    # Split data into Train (80%) and Eval (20%)
    random.shuffle(training_data)
    split_idx = int(len(training_data) * 0.8)
    train_data = training_data[:split_idx]
    eval_data = training_data[split_idx:]
    
    print(f"  Train samples: {len(train_data)} | Eval samples: {len(eval_data)}")
    print(f"  Iterations: {n_iter}")

    # Begin training
    optimizer = nlp.begin_training()

    # History tracking
    epoch_hist = []
    loss_hist = []
    f1_hist = []

    # Training loop
    for epoch in range(n_iter):
        random.shuffle(train_data)
        losses = {}

        # Create minibatches
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))

        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            nlp.update(examples, sgd=optimizer, drop=0.35, losses=losses)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Evaluate on dev set
            eval_examples = []
            for text, annotations in eval_data:
                doc = nlp.make_doc(text)
                eval_examples.append(Example.from_dict(doc, annotations))
            
            scores = nlp.evaluate(eval_examples)
            p = scores.get('ents_p', 0.0)
            r = scores.get('ents_r', 0.0)
            f = scores.get('ents_f', 0.0)
            print(f"  Epoch {epoch + 1:2d}/{n_iter} - Loss: {losses.get('ner', 0):.2f} "
                  f"- Eval => P: {p:.3f} | R: {r:.3f} | F1: {f:.3f}")
            epoch_hist.append(epoch + 1)
            loss_hist.append(losses.get('ner', 0))
            f1_hist.append(f)

    # Plot training history
    try:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss', color=color)
        ax1.plot(epoch_hist, loss_hist, color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Eval F1-Score', color=color)
        ax2.plot(epoch_hist, f1_hist, color=color, marker='s')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('NER Training: Loss vs F1-Score')
        fig.tight_layout()
        
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(output_dir)), "report_assets")
        os.makedirs(assets_dir, exist_ok=True)
        plot_path = os.path.join(assets_dir, "ner_training_history.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"  -> Saved training plot: {plot_path}")
    except ImportError:
        pass

    # Save the trained model
    os.makedirs(output_dir, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"  Model saved to: {output_dir}")

    return nlp


def load_trained_model(model_dir):
    """Load a previously trained spaCy NER model."""
    return spacy.load(model_dir)


# ─── Inference ───────────────────────────────────────────────

def recognize_entities(clause_text, nlp):
    """
    Run NER inference on a single clause.

    Args:
        clause_text: A single clause string.
        nlp:         A trained spaCy NER model.

    Returns:
        A dictionary with the clause and detected entities.
    """
    doc = nlp(clause_text.strip())

    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char
        })

    return {
        "clause": clause_text.strip(),
        "entities": entities
    }


# ─── Full pipeline ───────────────────────────────────────────

def train(data_dir, model_dir, n_iter=30):
    """
    Full training pipeline:
      1. Load training data
      2. Train spaCy NER model
      3. Save model

    Args:
        data_dir:  Path to directory containing ner_training_data.json
        model_dir: Path to save the trained model
        n_iter:    Number of training iterations
    """
    data_path = os.path.join(data_dir, "ner_training_data.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    training_data = load_training_data(data_path)
    nlp = train_ner_model(training_data, model_dir, n_iter=n_iter)
    return nlp


def process_file(input_path, output_path, model_dir=None):
    """
    Run NER inference on clauses using the trained model.

    Args:
        input_path:  Path to clauses file
        output_path: Path to output JSON
        model_dir:   Path to trained model directory
    """
    if model_dir and os.path.exists(model_dir):
        nlp = load_trained_model(model_dir)
        print(f"  Loaded trained model from: {model_dir}")
    else:
        raise FileNotFoundError(
            f"Trained NER model not found at: {model_dir}. "
            "Please run training first."
        )

    with open(input_path, "r", encoding="utf-8") as f:
        clauses = [line.strip() for line in f if line.strip()]

    results = []
    total_entities = 0

    for clause in clauses:
        result = recognize_entities(clause, nlp)
        results.append(result)
        total_entities += len(result["entities"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Processed {len(clauses)} clauses, "
          f"found {total_entities} entities.")
    print(f"  Output: {output_path}")
    return results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    model_dir = os.path.join(base_dir, "models", "ner_model")
    btl1_clauses = os.path.join(base_dir, "..", "BTL1", "output", "clauses.txt")
    output_path = os.path.join(base_dir, "output", "ner_results.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=== Training NER Model ===")
    train(data_dir, model_dir, n_iter=30)

    print("\n=== Running NER Inference ===")
    process_file(btl1_clauses, output_path, model_dir)
