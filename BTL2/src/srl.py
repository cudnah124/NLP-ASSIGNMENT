"""
Task 2.2: Semantic Role Labeling (SRL)
========================================
Uses AllenNLP's pre-trained BERT-based SRL model to extract semantic roles.

Roles: Agent (ARG0), Predicate (V), Theme (ARG1), Recipient (ARG2),
       Time (ARGM-TMP), Condition (ARGM-ADV), Location (ARGM-LOC)

If AllenNLP is not available, falls back to spaCy dependency-based extraction.

Input:  Clauses from Assignment 1 + NER results from Task 2.1
Output: output/srl_results.json
"""

import json
import os

# ─── Attempt to load AllenNLP SRL ────────────────────────────
_allennlp_predictor = None
_use_allennlp = False

try:
    from allennlp.predictors.predictor import Predictor
    import allennlp_models.structured_prediction
    _allennlp_predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/"
        "structured-prediction-srl-bert.2020.12.15.tar.gz"
    )
    _use_allennlp = True
    print("[SRL] Loaded AllenNLP BERT-based SRL model.")
except Exception:
    _use_allennlp = False
    print("[SRL] AllenNLP not available. Using spaCy dependency-based SRL.")


# ─── AllenNLP-based SRL ─────────────────────────────────────

# Map AllenNLP BIO tags to readable role names
ROLE_MAP = {
    "ARG0": "Agent",
    "ARG1": "Theme",
    "ARG2": "Recipient",
    "ARG3": "Beneficiary",
    "ARG4": "Attribute",
    "ARGM-TMP": "Time",
    "ARGM-LOC": "Location",
    "ARGM-MNR": "Manner",
    "ARGM-CAU": "Cause",
    "ARGM-PRP": "Purpose",
    "ARGM-ADV": "Condition",
    "ARGM-NEG": "Negation",
    "ARGM-MOD": "Modal",
    "ARGM-DIR": "Direction",
    "ARGM-EXT": "Extent",
    "V": "Predicate",
}


def extract_roles_allennlp(clause_text):
    """Extract semantic roles using AllenNLP's BERT-based SRL model."""
    prediction = _allennlp_predictor.predict(sentence=clause_text)
    words = prediction["words"]
    all_frames = []

    for verb_info in prediction["verbs"]:
        predicate = verb_info["verb"]
        tags = verb_info["tags"]
        roles = {}
        current_role = None
        current_tokens = []

        for word, tag in zip(words, tags):
            if tag.startswith("B-"):
                # Save previous role
                if current_role and current_tokens:
                    role_name = ROLE_MAP.get(current_role, current_role)
                    roles[role_name] = " ".join(current_tokens)
                current_role = tag[2:]
                current_tokens = [word]
            elif tag.startswith("I-") and current_role:
                current_tokens.append(word)
            else:
                if current_role and current_tokens:
                    role_name = ROLE_MAP.get(current_role, current_role)
                    roles[role_name] = " ".join(current_tokens)
                current_role = None
                current_tokens = []

        # Don't forget the last role
        if current_role and current_tokens:
            role_name = ROLE_MAP.get(current_role, current_role)
            roles[role_name] = " ".join(current_tokens)

        all_frames.append({
            "predicate": predicate,
            "roles": roles
        })

    return all_frames


# ─── spaCy dependency-based SRL (fallback) ───────────────────

def _load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


_spacy_nlp = None


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        _spacy_nlp = _load_spacy()
    return _spacy_nlp


def _get_phrase(token):
    tokens = [token]
    for child in token.children:
        if child.dep_ in ("compound", "flat", "amod", "det", "nummod", "poss"):
            tokens.extend(list(child.subtree))
    tokens = sorted(set(tokens), key=lambda t: t.i)
    return " ".join([t.text for t in tokens])


def _get_subtree(token):
    tokens = sorted(token.subtree, key=lambda t: t.i)
    return " ".join([t.text for t in tokens])


def extract_roles_spacy(clause_text):
    """Extract semantic roles using spaCy dependency parsing (fallback)."""
    nlp = _get_spacy()
    doc = nlp(clause_text.strip())
    roles = {}

    root = None
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
            break

    if root is None:
        return [{"predicate": None, "roles": {}}]

    # Predicate (with auxiliaries)
    aux = [c for c in root.children if c.dep_ in ("aux", "auxpass")]
    predicate = (" ".join(t.text for t in sorted(aux, key=lambda t: t.i))
                 + " " + root.text).strip()

    for child in root.children:
        dep = child.dep_
        if dep in ("nsubj",):
            roles["Agent"] = _get_phrase(child)
        elif dep == "nsubjpass":
            roles["Patient"] = _get_phrase(child)
        elif dep in ("dobj", "obj", "attr"):
            roles["Theme"] = _get_phrase(child)
        elif dep == "dative":
            roles["Recipient"] = _get_phrase(child)
        elif dep == "prep":
            pobj = next((gc for gc in child.children
                         if gc.dep_ in ("pobj", "obj")), None)
            if pobj:
                p = child.text.lower()
                obj_text = _get_phrase(pobj)
                if p in ("to", "unto") and "Recipient" not in roles:
                    roles["Recipient"] = obj_text
                elif p in ("before", "after", "by", "within", "during",
                           "until", "from", "on"):
                    roles.setdefault("Time", "")
                    roles["Time"] = (roles["Time"] + "; "
                                     + f"{p} {obj_text}").strip("; ")
        elif dep == "advcl":
            mark = next((gc.text.lower() for gc in child.children
                         if gc.dep_ == "mark"), None)
            if mark in ("if", "unless", "provided", "when", "should"):
                roles["Condition"] = _get_subtree(child)

    return [{"predicate": predicate, "roles": roles}]


# ─── Unified interface ───────────────────────────────────────

def extract_semantic_roles(clause_text):
    """
    Extract semantic roles from a clause.
    Uses AllenNLP BERT-SRL if available, else spaCy dependency fallback.
    """
    if _use_allennlp:
        frames = extract_roles_allennlp(clause_text)
    else:
        frames = extract_roles_spacy(clause_text)

    # Pick the primary frame (the one with most roles)
    if not frames:
        return {"clause": clause_text.strip(), "predicate": None, "roles": {}}

    best = max(frames, key=lambda f: len(f.get("roles", {})))
    return {
        "clause": clause_text.strip(),
        "predicate": best.get("predicate"),
        "roles": best.get("roles", {}),
        "all_frames": frames,
        "method": "allennlp_bert_srl" if _use_allennlp else "spacy_dependency"
    }


def process_file(input_path, output_path, ner_path=None):
    """Run SRL on all clauses and write results to JSON."""
    with open(input_path, "r", encoding="utf-8") as f:
        clauses = [line.strip() for line in f if line.strip()]

    # Optionally load NER results
    ner_results = None
    if ner_path and os.path.exists(ner_path):
        try:
            with open(ner_path, "r", encoding="utf-8") as f:
                ner_results = json.load(f)
        except (json.JSONDecodeError,):
            pass

    results = []
    for i, clause in enumerate(clauses):
        srl = extract_semantic_roles(clause)
        if ner_results and i < len(ner_results):
            srl["entities"] = ner_results[i].get("entities", [])
        results.append(srl)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    method = "AllenNLP BERT" if _use_allennlp else "spaCy dependency"
    roles_found = sum(1 for r in results if r["roles"])
    print(f"  Method: {method}")
    print(f"  Processed {len(clauses)} clauses, {roles_found} with roles.")
    print(f"  Output: {output_path}")
    return results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    btl1_clauses = os.path.join(base_dir, "..", "BTL1", "output", "clauses.txt")
    output_path = os.path.join(base_dir, "output", "srl_results.json")
    ner_path = os.path.join(base_dir, "output", "ner_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    process_file(btl1_clauses, output_path, ner_path)
