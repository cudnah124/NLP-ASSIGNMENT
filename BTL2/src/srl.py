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
import logging
import os
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="[SRL] %(message)s")
logger = logging.getLogger(__name__)

# ─── Attempt to load AllenNLP SRL ────────────────────────────

_allennlp_predictor = None
_use_allennlp = False

try:
    from allennlp.predictors.predictor import Predictor
    import allennlp_models.structured_prediction  # noqa: F401  (registers models)

    _allennlp_predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/"
        "structured-prediction-srl-bert.2020.12.15.tar.gz"
    )
    _use_allennlp = True
    logger.info("Loaded AllenNLP BERT-based SRL model.")
except Exception:
    logger.info("AllenNLP not available. Using spaCy dependency-based SRL.")


# ─── Role mapping ────────────────────────────────────────────

# Maps AllenNLP BIO argument tags to human-readable role names.
# NOTE: "V" (predicate) is intentionally excluded here because it is
# already captured as the top-level `predicate` field in each frame.
ROLE_MAP: dict[str, str] = {
    "ARG0":     "Agent",
    "ARG1":     "Theme",
    "ARG2":     "Recipient",
    "ARG3":     "Beneficiary",
    "ARG4":     "Attribute",
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
}


# ─── AllenNLP-based SRL ──────────────────────────────────────

def extract_roles_allennlp(clause_text: str) -> list[dict[str, Any]]:
    """Extract semantic roles using AllenNLP's BERT-based SRL model."""
    prediction = _allennlp_predictor.predict(sentence=clause_text)
    words: list[str] = prediction["words"]
    all_frames: list[dict[str, Any]] = []

    for verb_info in prediction["verbs"]:
        predicate: str = verb_info["verb"]
        tags: list[str] = verb_info["tags"]
        roles: dict[str, str] = {}
        current_tag: Optional[str] = None
        current_tokens: list[str] = []

        def _flush_role() -> None:
            """Save the current accumulated role span into `roles`."""
            if current_tag and current_tokens:
                # Skip "V" — already stored as `predicate`
                if current_tag != "V":
                    role_name = ROLE_MAP.get(current_tag, current_tag)
                    roles[role_name] = " ".join(current_tokens)

        for word, tag in zip(words, tags):
            if tag.startswith("B-"):
                _flush_role()
                current_tag = tag[2:]
                current_tokens = [word]
            elif tag.startswith("I-") and current_tag:
                current_tokens.append(word)
            else:
                _flush_role()
                current_tag = None
                current_tokens = []

        _flush_role()  # flush the last span

        all_frames.append({"predicate": predicate, "roles": roles})

    return all_frames


# ─── spaCy dependency-based SRL (fallback) ───────────────────

_spacy_nlp = None


def _get_spacy():
    """Lazy-load spaCy model, downloading if necessary."""
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp


def _get_phrase(token) -> str:
    """
    Return the surface form of a token together with its close dependents
    (determiners, modifiers, compounds, etc.), preserving original order.
    """
    MODIFIER_DEPS = {"compound", "flat", "amod", "det", "nummod", "poss"}
    # Collect token indices to avoid duplicates while preserving order
    seen: set[int] = set()
    collected = []

    def _collect(t) -> None:
        if t.i in seen:
            return
        seen.add(t.i)
        collected.append(t)

    _collect(token)
    for child in token.children:
        if child.dep_ in MODIFIER_DEPS:
            for t in child.subtree:
                _collect(t)

    return " ".join(t.text for t in sorted(collected, key=lambda t: t.i))


def _get_subtree(token) -> str:
    """Return the full text of a token's subtree in sentence order."""
    return " ".join(t.text for t in sorted(token.subtree, key=lambda t: t.i))


def extract_roles_spacy(clause_text: str) -> list[dict[str, Any]]:
    """Extract semantic roles using spaCy dependency parsing (fallback)."""
    nlp = _get_spacy()
    doc = nlp(clause_text.strip())

    root = next((token for token in doc if token.dep_ == "ROOT"), None)
    if root is None:
        return [{"predicate": None, "roles": {}}]

    # Build predicate string: auxiliaries + root verb
    auxiliaries = sorted(
        (c for c in root.children if c.dep_ in ("aux", "auxpass")),
        key=lambda t: t.i,
    )
    predicate = " ".join(t.text for t in auxiliaries + [root]).strip()

    roles: dict[str, str] = {}
    time_parts: list[str] = []

    TIME_PREPS = {"before", "after", "by", "within", "during", "until", "from", "on"}
    CONDITION_MARKERS = {"if", "unless", "provided", "when", "should"}

    for child in root.children:
        dep = child.dep_

        if dep == "nsubj":
            roles["Agent"] = _get_phrase(child)

        elif dep == "nsubjpass":
            roles["Patient"] = _get_phrase(child)

        elif dep in ("dobj", "obj", "attr"):
            roles["Theme"] = _get_phrase(child)

        elif dep == "dative":
            roles["Recipient"] = _get_phrase(child)

        elif dep == "prep":
            pobj = next(
                (gc for gc in child.children if gc.dep_ in ("pobj", "obj")),
                None,
            )
            if pobj:
                prep_text = child.text.lower()
                obj_text = _get_phrase(pobj)

                if prep_text in ("to", "unto") and "Recipient" not in roles:
                    roles["Recipient"] = obj_text
                elif prep_text in TIME_PREPS:
                    time_parts.append(f"{prep_text} {obj_text}")

        elif dep == "advcl":
            mark = next(
                (gc.text.lower() for gc in child.children if gc.dep_ == "mark"),
                None,
            )
            if mark in CONDITION_MARKERS:
                roles["Condition"] = _get_subtree(child)

    if time_parts:
        roles["Time"] = "; ".join(time_parts)

    return [{"predicate": predicate, "roles": roles}]


# ─── Unified interface ───────────────────────────────────────

def extract_semantic_roles(clause_text: str) -> dict[str, Any]:
    """
    Extract semantic roles from a single clause.

    Uses AllenNLP BERT-SRL when available, otherwise falls back to
    spaCy dependency-based extraction.

    Returns a dict with keys:
        clause     – original clause text
        predicate  – main predicate string
        roles      – dict of role_name → span
        all_frames – all verb frames found (AllenNLP may find multiple)
        method     – which backend was used
    """
    frames: list[dict[str, Any]] = (
        extract_roles_allennlp(clause_text)
        if _use_allennlp
        else extract_roles_spacy(clause_text)
    )

    if not frames:
        return {
            "clause": clause_text.strip(),
            "predicate": None,
            "roles": {},
            "all_frames": [],
            "method": "allennlp_bert_srl" if _use_allennlp else "spacy_dependency",
        }

    # Choose the frame that has the most roles as the primary one
    best = max(frames, key=lambda f: len(f.get("roles", {})))

    return {
        "clause": clause_text.strip(),
        "predicate": best.get("predicate"),
        "roles": best.get("roles", {}),
        "all_frames": frames,
        "method": "allennlp_bert_srl" if _use_allennlp else "spacy_dependency",
    }


# ─── File-level processing ───────────────────────────────────

def _load_ner_results(ner_path: Optional[str]) -> Optional[list[dict]]:
    """Load NER results from JSON, returning None if unavailable."""
    if not ner_path or not os.path.exists(ner_path):
        return None
    try:
        with open(ner_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Could not parse NER file '%s': %s", ner_path, exc)
        return None


def process_file(
    input_path: str,
    output_path: str,
    ner_path: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Run SRL on every clause in `input_path` and write results to `output_path`.

    Args:
        input_path:  Path to a plain-text file with one clause per line.
        output_path: Destination JSON file for SRL results.
        ner_path:    Optional path to NER results JSON (Task 2.1 output).

    Returns:
        List of SRL result dicts.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            clauses = [line.strip() for line in f if line.strip()]
    except OSError as exc:
        logger.error("Cannot read input file '%s': %s", input_path, exc)
        raise

    ner_results = _load_ner_results(ner_path)

    results: list[dict[str, Any]] = []
    for i, clause in enumerate(clauses):
        srl = extract_semantic_roles(clause)
        if ner_results is not None and i < len(ner_results):
            srl["entities"] = ner_results[i].get("entities", [])
        results.append(srl)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    method_label = "AllenNLP BERT" if _use_allennlp else "spaCy dependency"
    roles_found = sum(1 for r in results if r["roles"])
    logger.info("Method    : %s", method_label)
    logger.info("Clauses   : %d total, %d with roles", len(clauses), roles_found)
    logger.info("Output    : %s", output_path)

    return results


# ─── Entry point ─────────────────────────────────────────────

if __name__ == "__main__":
    this_file   = Path(__file__).resolve()
    btl2_dir    = this_file.parent.parent         
    project_dir = btl2_dir.parent                  

    btl1_clauses = project_dir / "BTL1" / "output" / "clauses.txt"
    output_path  = btl2_dir   / "output" / "srl_results.json"
    ner_path     = btl2_dir   / "output" / "ner_results.json"

    process_file(str(btl1_clauses), str(output_path), str(ner_path))