"""
Task 2.3: Clause Intent Classification
=========================================
Classify each clause into: Obligation, Prohibition, Right, or Termination Condition.

Two approaches (trained and compared):
  1. Baseline:  TF-IDF + Logistic Regression (scikit-learn)
  2. Advanced:  Fine-tuned DistilBERT transformer (Hugging Face)

Training data:  data/intent_training_data.json
Input:          Clauses from Assignment 1
Output:         output/intent_classification.txt
                output/intent_classification.json
"""

import json
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ─── Label definitions ───────────────────────────────────────
LABELS = ["Obligation", "Prohibition", "Right", "Termination Condition"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


# ─── Data loading ────────────────────────────────────────────

def load_training_data(data_path):
    """Load labeled training data from JSON."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels


# ══════════════════════════════════════════════════════════════
# BASELINE: TF-IDF + Logistic Regression
# ══════════════════════════════════════════════════════════════

def train_tfidf_model(texts, labels, model_dir):
    """
    Train a TF-IDF + Logistic Regression classifier.

    Args:
        texts:     List of clause strings.
        labels:    List of intent label strings.
        model_dir: Path to save the trained model.

    Returns:
        (vectorizer, classifier) tuple.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    import joblib

    print("  Training TF-IDF + Logistic Regression...")

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    X = vectorizer.fit_transform(texts)
    y = np.array([LABEL2ID[label] for label in labels])

    # Train Logistic Regression
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X, y)

    # Cross-validation evaluation
    scores = cross_val_score(clf, X, y, cv=min(5, len(texts)), scoring="accuracy")
    print(f"  Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(clf, os.path.join(model_dir, "logreg_classifier.pkl"))
    print(f"  Model saved to: {model_dir}")

    return vectorizer, clf


def predict_tfidf(texts, model_dir):
    """Run inference with the trained TF-IDF + LogReg model."""
    import joblib
    vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    clf = joblib.load(os.path.join(model_dir, "logreg_classifier.pkl"))

    X = vectorizer.transform(texts)
    pred_ids = clf.predict(X)
    probas = clf.predict_proba(X)

    results = []
    for i, text in enumerate(texts):
        label = ID2LABEL[pred_ids[i]]
        confidence = float(probas[i].max())
        results.append({
            "clause": text,
            "intent": label,
            "confidence": round(confidence, 4),
            "method": "tfidf_logreg"
        })
    return results


# ══════════════════════════════════════════════════════════════
# ADVANCED: Fine-tuned DistilBERT
# ══════════════════════════════════════════════════════════════

def train_transformer_model(texts, labels, model_dir, epochs=5):
    """
    Fine-tune a DistilBERT model for intent classification.

    Args:
        texts:     List of clause strings.
        labels:    List of intent label strings.
        model_dir: Path to save the fine-tuned model.
        epochs:    Number of training epochs.

    Returns:
        True if training succeeded, False otherwise.
    """
    try:
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertForSequenceClassification,
            Trainer,
            TrainingArguments,
        )
        import torch
        from torch.utils.data import Dataset
    except ImportError:
        print("  [SKIP] transformers/torch not installed. "
              "Install with: pip install transformers torch")
        return False

    print("  Fine-tuning DistilBERT for intent classification...")

    # Tokenize
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    label_ids = [LABEL2ID[l] for l in labels]

    # Dataset class
    class IntentDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx])
                    for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    dataset = IntentDataset(encodings, label_ids)

    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(model_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        no_cuda=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # Save model and tokenizer
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"  DistilBERT model saved to: {model_dir}")
    return True


def predict_transformer(texts, model_dir):
    """Run inference with the fine-tuned DistilBERT model."""
    try:
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertForSequenceClassification,
        )
        import torch
    except ImportError:
        return None

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probas = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(probas, dim=-1).item()
        confidence = probas[0][pred_id].item()

        results.append({
            "clause": text,
            "intent": ID2LABEL[pred_id],
            "confidence": round(confidence, 4),
            "method": "distilbert"
        })

    return results


# ══════════════════════════════════════════════════════════════
# Full pipeline
# ══════════════════════════════════════════════════════════════

def train(data_dir, model_dir, train_transformer=True):
    """
    Train both baseline and advanced intent classification models.

    Args:
        data_dir:          Path to data directory with intent_training_data.json
        model_dir:         Path to save models
        train_transformer: Whether to also train the DistilBERT model
    """
    data_path = os.path.join(data_dir, "intent_training_data.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    texts, labels = load_training_data(data_path)
    print(f"  Loaded {len(texts)} training samples")
    print(f"  Label distribution: { {l: labels.count(l) for l in LABELS} }")

    # Train baseline
    tfidf_dir = os.path.join(model_dir, "tfidf_logreg")
    train_tfidf_model(texts, labels, tfidf_dir)

    # Train advanced (optional)
    if train_transformer:
        transformer_dir = os.path.join(model_dir, "distilbert")
        success = train_transformer_model(texts, labels, transformer_dir, epochs=5)
        if not success:
            print("  Transformer training skipped.")


def process_file(input_path, output_path, model_dir=None):
    """
    Run intent classification inference and write results.
    Uses both baseline and advanced models, comparing their outputs.

    Args:
        input_path:  Path to clauses file
        output_path: Path to output text file
        model_dir:   Path to trained models directory
    """
    with open(input_path, "r", encoding="utf-8") as f:
        clauses = [line.strip() for line in f if line.strip()]

    # Run baseline (TF-IDF + LogReg)
    tfidf_dir = os.path.join(model_dir, "tfidf_logreg")
    baseline_results = predict_tfidf(clauses, tfidf_dir)

    # Try advanced (DistilBERT)
    transformer_dir = os.path.join(model_dir, "distilbert")
    advanced_results = None
    if os.path.exists(transformer_dir):
        advanced_results = predict_transformer(clauses, transformer_dir)

    # Merge results: prefer transformer if available, else use baseline
    final_results = []
    for i, clause in enumerate(clauses):
        baseline = baseline_results[i]
        if advanced_results:
            advanced = advanced_results[i]
            final_results.append({
                "clause": clause,
                "intent": advanced["intent"],
                "confidence": advanced["confidence"],
                "method": "distilbert",
                "baseline_intent": baseline["intent"],
                "baseline_confidence": baseline["confidence"],
            })
        else:
            final_results.append(baseline)

    # Count intents
    intent_counts = {l: 0 for l in LABELS}
    for r in final_results:
        intent_counts[r["intent"]] += 1

    # Write text output (clause + label)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in final_results:
            f.write(f"{r['clause']}\t{r['intent']}\n")

    # Write detailed JSON
    json_path = output_path.replace(".txt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Print comparison if both models ran
    if advanced_results:
        agree = sum(1 for b, a in zip(baseline_results, advanced_results)
                    if b["intent"] == a["intent"])
        print(f"  Model agreement: {agree}/{len(clauses)} "
              f"({agree/len(clauses)*100:.1f}%)")

    print(f"  Distribution: {dict(intent_counts)}")
    print(f"  Output: {output_path}")
    return final_results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    model_dir = os.path.join(base_dir, "models", "intent")
    btl1_clauses = os.path.join(base_dir, "..", "BTL1", "output", "clauses.txt")
    output_path = os.path.join(base_dir, "output", "intent_classification.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=== Training Intent Models ===")
    train(data_dir, model_dir, train_transformer=True)

    print("\n=== Running Intent Classification ===")
    process_file(btl1_clauses, output_path, model_dir)
