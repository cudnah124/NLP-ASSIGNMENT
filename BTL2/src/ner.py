import json
import os
import random
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

ENTITY_LABELS = {"PARTY", "MONEY", "DATE", "RATE", "PENALTY", "LAW"}
def load_training_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    training_data = []
    skipped_labels = set()

    for item in raw_data:
        text = item["text"]
        raw_entities = [tuple(e) for e in item["entities"]]

        valid_entities = []
        for start, end, label in raw_entities:
            if label not in ENTITY_LABELS:
                skipped_labels.add(label)
                continue
            valid_entities.append((start, end, label))
        valid_entities = _remove_overlaps(valid_entities)

        if valid_entities:
            training_data.append((text, {"entities": valid_entities}))

    return training_data


def _remove_overlaps(entities):
    sorted_ents = sorted(entities, key=lambda e: (e[0], e[1]))
    result = []
    last_end = -1
    for start, end, label in sorted_ents:
        if start >= last_end:
            result.append((start, end, label))
            last_end = end
    return result

def train_ner_model(training_data, output_dir, n_iter=30):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner", last=True)

    labels = set()
    for _, annotations in training_data:
        for ent in annotations["entities"]:
            labels.add(ent[2])
    for label in sorted(labels):
        ner.add_label(label)
    random.shuffle(training_data)
    split_idx = int(len(training_data) * 0.8)
    train_data = training_data[:split_idx]
    eval_data  = training_data[split_idx:]

    optimizer = nlp.begin_training()

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

            epoch_hist.append(epoch + 1)
            loss_hist.append(losses.get("ner", 0))
            f1_hist.append(overall_f1)

    _save_training_plot(epoch_hist, loss_hist, f1_hist, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    nlp.to_disk(output_dir)
    return nlp


def _save_training_plot(epoch_hist, loss_hist, f1_hist, model_output_dir):
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
    except ImportError:
        pass


def load_trained_model(model_dir):
    return spacy.load(model_dir)
def recognize_entities(clause_text, nlp):
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
def train(data_dir, model_dir, n_iter=30):
    data_path = os.path.join(data_dir, "ner_training_data.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")
    training_data = load_training_data(data_path)
    return train_ner_model(training_data, model_dir, n_iter=n_iter)


def process_file(input_path, output_path, model_dir=None):
    if not (model_dir and os.path.exists(model_dir)):
        raise FileNotFoundError(
            f"Trained NER model not found at: {model_dir}\n"
            "  Please run training first (--mode train)."
        )
    nlp = load_trained_model(model_dir)

    with open(input_path, "r", encoding="utf-8") as f:
        clauses = [line.strip() for line in f if line.strip()]
    results = []
    for clause in clauses:
        result = recognize_entities(clause, nlp)
        results.append(result)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results
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
        train(data_dir, model_dir, n_iter=args.iter)

    if args.mode in ("infer", "all"):
        process_file(clauses_in, output_path, model_dir)