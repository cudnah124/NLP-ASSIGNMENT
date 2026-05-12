import json
import os
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

LABELS = ["Obligation", "Prohibition", "Right", "Termination Condition"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
def load_training_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels


def train_tfidf_model(texts, labels, model_dir):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    import joblib

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    X = vectorizer.fit_transform(texts)
    y = np.array([LABEL2ID[label] for label in labels])

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X, y)
    scores = cross_val_score(clf, X, y, cv=min(5, len(texts)), scoring="accuracy")

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(clf, os.path.join(model_dir, "logreg_classifier.pkl"))
    return vectorizer, clf


def predict_tfidf(texts, model_dir):
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


def train_transformer_model(texts, labels, model_dir, epochs=5):
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
        return False

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    label_ids = [LABEL2ID[l] for l in labels]

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

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    training_args = TrainingArguments(
        output_dir=os.path.join(model_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    
    # Evaluation
    print("\n" + "="*50)
    print("TRANSFORMER MODEL (DistilBERT) EVALUATION")
    print("="*50)
    
    metrics = trainer.evaluate(eval_dataset=dataset)
    print(f"Evaluation Metrics: {metrics}")

    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return True


def predict_transformer(texts, model_dir):
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


def train(data_dir, model_dir, train_transformer=True):
    data_path = os.path.join(data_dir, "intent_training_data.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")
    texts, labels = load_training_data(data_path)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    tfidf_dir = os.path.join(model_dir, "tfidf_logreg")
    train_tfidf_model(train_texts, train_labels, tfidf_dir)
    
    # Evaluate TF-IDF
    tfidf_preds = predict_tfidf(test_texts, tfidf_dir)
    y_pred_tfidf = [p["intent"] for p in tfidf_preds]
    
    print("\n" + "="*50)
    print("TF-IDF + LOGISTIC REGRESSION EVALUATION")
    print("="*50)
    print(classification_report(test_labels, y_pred_tfidf))
    plot_confusion_matrix(test_labels, y_pred_tfidf, LABELS, "Confusion Matrix: TF-IDF + LogReg", 
                          os.path.join(os.path.dirname(model_dir), "..", "report_assets", "intent_confusion_matrix.png"))

    if train_transformer:
        transformer_dir = os.path.join(model_dir, "distilbert")
        success = train_transformer_model(train_texts, train_labels, transformer_dir, epochs=5)
        
        if success:
            transformer_preds = predict_transformer(test_texts, transformer_dir)
            y_pred_trans = [p["intent"] for p in transformer_preds]
            print("\n" + "="*50)
            print("TRANSFORMER (DistilBERT) FINAL TEST EVALUATION")
            print("="*50)
            print(classification_report(test_labels, y_pred_trans))
            plot_confusion_matrix(test_labels, y_pred_trans, LABELS, "Confusion Matrix: DistilBERT", 
                                  os.path.join(os.path.dirname(model_dir), "..", "report_assets", "intent_confusion_matrix_distilbert.png"))
        else:
            print("Transformer training skipped or failed.")

def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    except Exception as e:
        print(f"Failed to plot confusion matrix: {e}")


def process_file(input_path, output_path, model_dir=None):
    with open(input_path, "r", encoding="utf-8") as f:
        clauses = [line.strip() for line in f if line.strip()]
    tfidf_dir = os.path.join(model_dir, "tfidf_logreg")
    baseline_results = predict_tfidf(clauses, tfidf_dir)
    transformer_dir = os.path.join(model_dir, "distilbert")
    advanced_results = None
    if os.path.exists(transformer_dir):
        advanced_results = predict_transformer(clauses, transformer_dir)
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
    with open(output_path, "w", encoding="utf-8") as f:
        for r in final_results:
            f.write(f"{r['clause']}\t{r['intent']}\n")
    json_path = output_path.replace(".txt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    return final_results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    model_dir = os.path.join(base_dir, "models", "intent")
    btl1_clauses = os.path.join(base_dir, "..", "BTL1", "output", "clauses.txt")
    output_path = os.path.join(base_dir, "output", "intent_classification.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    train(data_dir, model_dir, train_transformer=True)
    process_file(btl1_clauses, output_path, model_dir)
