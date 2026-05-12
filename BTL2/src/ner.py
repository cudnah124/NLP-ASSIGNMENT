import json
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

# 1. Config labels
ENTITY_LABELS = ["O", "PARTY", "MONEY", "DATE", "RATE", "PENALTY", "LAW"]
LABEL2ID = {label: i for i, label in enumerate(ENTITY_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(ENTITY_LABELS)}

MODEL_CHECKPOINT = "dslim/bert-base-NER"
MAX_LEN = 128

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=MAX_LEN):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item["text"]
        entities = item["entities"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        labels = [LABEL2ID["O"]] * self.max_len
        offsets = encoding["offset_mapping"][0]
        
        for start, end, label in entities:
            label_id = LABEL2ID.get(label, LABEL2ID["O"])
            for idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == 0 and tok_end == 0:
                    continue
                # Align character offsets to tokens
                if tok_start >= start and tok_end <= end:
                    labels[idx] = label_id

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def load_training_data(data_path):
    if not os.path.exists(data_path):
        return []
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

def train_ner_model(training_data, output_dir, n_iter=10):
    print(f"Loading base model: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, 
        num_labels=len(ENTITY_LABELS),
        ignore_mismatched_sizes=True
    )

    # Force requires_grad=True for ALL parameters.
    # When ignore_mismatched_sizes=True reinit the classifier layer, some
    # transformers versions create those new tensors without grad, which
    # breaks the autograd graph and causes "does not require grad" at loss.backward().
    for param in model.parameters():
        param.requires_grad_(True)

    random.shuffle(training_data)

    # --- Stratified split by label signature ---
    # Group samples by which labels they contain, then distribute each group
    # proportionally → rare labels (PENALTY, RATE) appear in all splits.
    from collections import defaultdict
    groups = defaultdict(list)
    for sample in training_data:
        sig = frozenset(l for _, _, l in sample.get("entities", []))
        groups[sig].append(sample)

    train_data, val_data, test_data = [], [], []
    for sig, samples in groups.items():
        random.shuffle(samples)
        n_s = len(samples)
        t1  = max(1, int(n_s * 0.8))
        t2  = max(t1, int(n_s * 0.9))
        train_data.extend(samples[:t1])
        val_data.extend(samples[t1:t2])
        test_data.extend(samples[t2:])

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    # --- End stratified split ---

    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    train_dataset = NERDataset(train_data, tokenizer)
    val_dataset = NERDataset(val_data, tokenizer)
    test_dataset = NERDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()  # Ensure model is in training mode after loading

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    total_steps = len(train_loader) * n_iter
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    epoch_hist, loss_hist, val_loss_hist = [], [], []

    # Early stopping state
    patience        = 5
    best_val_loss   = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(output_dir, "best_ckpt")

    print(f"Training on {device} for {n_iter} epochs (patience={patience})...")
    torch.set_grad_enabled(True)  # Safety: re-enable in case any import disabled it globally
    for epoch in range(n_iter):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if loss is None:
                continue  # Skip batch if loss is None (label alignment issue)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)

        # Always record history (before potential break)
        epoch_hist.append(epoch + 1)
        loss_hist.append(avg_train_loss)
        val_loss_hist.append(avg_val_loss)

        # Early stopping check — decide BEFORE printing so we can print a complete line
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            patience_counter = 0
            tag = "✅ best"
        else:
            patience_counter += 1
            tag = f"no improvement {patience_counter}/{patience}"

        print(f"Epoch {epoch+1}/{n_iter} - Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} [{tag}]")

        # Save checkpoint AFTER printing (avoids 'Writing model shards' interrupting the log)
        if is_best:
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} — no improvement for {patience} epochs.")
            break

    # Restore best checkpoint before saving final model
    if os.path.exists(best_model_path):
        print(f"Restoring best model (val_loss={best_val_loss:.4f})...")
        model = type(model).from_pretrained(best_model_path)
        model.to(device)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    with open(os.path.join(output_dir, "labels.json"), "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f)

    _save_training_plot(epoch_hist, loss_hist, val_loss_hist, output_dir)
    
    # Evaluate on Test Set
    print("\nEvaluating on Test Set...")
    evaluate_ner_model(model, tokenizer, test_loader, test_data, output_dir)
    
    return model, tokenizer

def evaluate_ner_model(model, tokenizer, test_loader, test_raw_data, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=2)
            
            # Mask out padding
            for i in range(labels.shape[0]):
                mask = attention_mask[i] == 1
                all_preds.extend(preds[i][mask].cpu().numpy())
                all_labels.extend(labels[i][mask].cpu().numpy())
    
    # Filter out special labels if any (though here we use all ENTITY_LABELS)
    target_names = [ENTITY_LABELS[i] for i in range(len(ENTITY_LABELS))]
    report = classification_report(all_labels, all_preds, target_names=target_names, labels=range(len(ENTITY_LABELS)))
    print(report)
    
    # Save report
    with open(os.path.join(output_dir, "test_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    
    # Save Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds, ENTITY_LABELS, "Confusion Matrix: BERT NER", 
                          os.path.join(os.path.dirname(os.path.dirname(output_dir)), "report_assets", "ner_confusion_matrix.png"))

def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        # Convert numeric IDs back to labels if needed
        y_true_labels = [ID2LABEL[i] if isinstance(i, (int, np.integer)) else i for i in y_true]
        y_pred_labels = [ID2LABEL[i] if isinstance(i, (int, np.integer)) else i for i in y_pred]
        
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    except Exception as e:
        print(f"Failed to plot confusion matrix: {e}")
    
    # Error Analysis: Compare predicted entities vs ground truth for each test sample
    error_analysis = []
    for item in test_raw_data:
        text = item["text"]
        gt_entities = item["entities"] # list of [start, end, label]
        
        pred_result = recognize_entities(text, model, tokenizer)
        pred_entities = pred_result["entities"] # list of dicts
        
        # Convert predictions to [start, end, label] for comparison
        pred_triples = [[e["start"], e["end"], e["label"]] for e in pred_entities]
        
        # Sort both for comparison
        gt_sorted = sorted(gt_entities, key=lambda x: x[0])
        pred_sorted = sorted(pred_triples, key=lambda x: x[0])
        
        is_match = (gt_sorted == pred_sorted)
        
        error_analysis.append({
            "text": text,
            "ground_truth": gt_entities,
            "predictions": pred_entities,
            "is_match": is_match
        })
    
    error_file = os.path.join(output_dir, "test_error_analysis.json")
    with open(error_file, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"Error analysis saved to {error_file}")

def _save_training_plot(epoch_hist, loss_hist, val_loss_hist, model_output_dir):
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_hist, loss_hist, label='Train Loss', color='tab:red', marker='o')
        plt.plot(epoch_hist, val_loss_hist, label='Val Loss', color='tab:blue', marker='s')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('BERT NER Training: Loss History')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(model_output_dir)), "report_assets")
        os.makedirs(assets_dir, exist_ok=True)
        plt.savefig(os.path.join(assets_dir, "ner_training_history.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

def recognize_entities(text, model, tokenizer):
    device = next(model.parameters()).device
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=MAX_LEN, 
        return_offsets_mapping=True,
        padding=True
    ).to(device)
    
    offsets = inputs.pop("offset_mapping")[0].cpu().numpy()
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    
    entities = []
    current_ent = None

    # Get word_ids to handle sub-tokens correctly
    word_ids = inputs.word_ids(batch_index=0)

    for idx, label_id in enumerate(predictions):
        label = ID2LABEL[label_id]
        start, end = offsets[idx]
        
        # Skip special tokens
        if word_ids[idx] is None:
            continue
            
        if label == "O":
            if current_ent:
                entities.append(current_ent)
                current_ent = None
            continue

        if current_ent and current_ent["label"] == label:
            # Check if this is a continuation of the same word or consecutive word
            # Just extend the end offset
            current_ent["end"] = int(end)
            current_ent["text"] = text[current_ent["start"]:current_ent["end"]]
        else:
            if current_ent:
                entities.append(current_ent)
            
            current_ent = {
                "text": text[start:end],
                "label": label,
                "start": int(start),
                "end": int(end)
            }
            
    if current_ent:
        entities.append(current_ent)
            
    return {"clause": text, "entities": entities}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

def train(data_dir, model_dir, n_iter=10):
    data_path = os.path.join(data_dir, "ner_training_data.json")
    training_data = load_training_data(data_path)
    if not training_data:
        print("No training data found.")
        return None, None
    return train_ner_model(training_data, model_dir, n_iter=n_iter)

def process_file(input_path, output_path, model_dir):
    if not os.path.exists(model_dir):
        print(f"Model not found at {model_dir}")
        return
    
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with open(input_path, "r", encoding="utf-8") as f:
        clauses = [line.strip() for line in f if line.strip()]
    
    results = []
    for clause in clauses:
        results.append(recognize_entities(clause, model, tokenizer))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer", "all"], default="all")
    parser.add_argument("--iter", type=int, default=10)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    model_dir = os.path.join(base_dir, "models", "ner_model")
    input_txt = os.path.join(base_dir, "input", "clauses.txt")
    output_json = os.path.join(base_dir, "output", "ner_results.json")

    if args.mode in ["train", "all"]:
        train(data_dir, model_dir, n_iter=args.iter)
    
    if args.mode in ["infer", "all"]:
        process_file(input_txt, output_json, model_dir)