import json
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup

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

    random.shuffle(training_data)
    split = int(len(training_data) * 0.8)
    train_data = training_data[:split]
    val_data = training_data[split:]

    train_dataset = NERDataset(train_data, tokenizer)
    val_dataset = NERDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    total_steps = len(train_loader) * n_iter
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    epoch_hist, loss_hist, val_loss_hist = [], [], []

    print(f"Training on {device} for {n_iter} epochs...")
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
        print(f"Epoch {epoch+1}/{n_iter} - Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        epoch_hist.append(epoch + 1)
        loss_hist.append(avg_train_loss)
        val_loss_hist.append(avg_val_loss)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    with open(os.path.join(output_dir, "labels.json"), "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f)

    _save_training_plot(epoch_hist, loss_hist, val_loss_hist, output_dir)
    return model, tokenizer

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
        return_offsets_mapping=True
    ).to(device)
    
    offsets = inputs.pop("offset_mapping")[0].cpu().numpy()
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    
    entities = []
    current_ent = None

    for idx, label_id in enumerate(predictions):
        label = ID2LABEL[label_id]
        start, end = offsets[idx]
        
        # Bỏ qua special tokens và nhãn "O"
        if start == end or label == "O":
            if current_ent:
                entities.append(current_ent)
                current_ent = None
            continue

        if current_ent and current_ent["label"] == label:
            # Gộp vào entity hiện tại
            current_ent["text"] = text[current_ent["start"]:end]
            current_ent["end"] = int(end.item())
        else:
            # Lưu entity cũ nếu có
            if current_ent:
                entities.append(current_ent)
            
            # Tạo entity mới
            current_ent = {
                "text": text[start:end],
                "label": label,
                "start": int(start.item()),
                "end": int(end.item())
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