"""
Task 2.2: Semantic Role Labeling (SRL)
========================================
Uses yeomtong/srl_bert_model for BERT-based predicate-aware SRL.

Model is downloaded from HuggingFace Hub, then loaded via the repo's
own `predictor` / `model` modules.  Output of `prediction_formatted()`
mirrors the AllenNLP schema:

    {
      "words": ["I", "want", ...],
      "verbs": [
        {"verb": "want", "tags": ["B-ARG0", "B-V", "B-ARG1", ...], ...},
        ...
      ]
    }

Falls back to spaCy dependency parsing if the model cannot be loaded.

Input:  Clauses from Assignment 1  +  NER results from Task 2.1
Output: output/srl_results.json
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="[SRL] %(message)s")
logger = logging.getLogger(__name__)

# ─── Model identifier ────────────────────────────────────────

HF_REPO_ID   = "yeomtong/srl_bert_model"
HF_CKPT_FILE = "best_srl_Sep_29.ckpt"
BERT_NAME    = "bert-base-cased"

# ─── Load yeomtong/srl_bert_model ────────────────────────────

_prediction_formatted = None   # callable: str -> dict
_use_hf = False

try:
    from huggingface_hub import hf_hub_download, snapshot_download

    logger.info("Downloading checkpoint …")
    ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_CKPT_FILE)

    logger.info("Downloading repo sources …")
    repo_dir = snapshot_download(HF_REPO_ID)
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

    from predictor import srl_init
    from visualizer import prediction_formatted

    logger.info("Initialising SRL model (bert-base-cased) …")
    srl_init(ckpt_path, bert_name=BERT_NAME)

    _prediction_formatted = prediction_formatted
    _use_hf = True
    logger.info("yeomtong/srl_bert_model loaded successfully.")

except Exception as exc:
    logger.warning("Model unavailable (%s). Falling back to spaCy.", exc)


# ─── PropBank → readable role name mapping ───────────────────

# "V" is intentionally absent: the verb is stored as `predicate`, not in roles.
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


# ─── HuggingFace BERT-based SRL ──────────────────────────────

def _decode_bio_spans(words: list[str], tags: list[str]) -> dict[str, str]:
    """
    Convert a BIO tag sequence into {role_name: span_text}.

    Handles B-LABEL / I-LABEL scheme.
    The "V" tag (predicate) is skipped — stored separately as `predicate`.
    """
    roles: dict[str, str] = {}
    current_tag: Optional[str] = None
    current_tokens: list[str] = []

    def _flush() -> None:
        if current_tag and current_tokens and current_tag != "V":
            role_name = ROLE_MAP.get(current_tag, current_tag)
            roles[role_name] = " ".join(current_tokens)

    for word, tag in zip(words, tags):
        if tag == "O":
            _flush()
            current_tag, current_tokens = None, []
        elif tag.startswith("B-"):
            _flush()
            current_tag  = tag[2:]
            current_tokens = [word]
        elif tag.startswith("I-") and current_tag == tag[2:]:
            current_tokens.append(word)
        else:
            # Unexpected I- without matching B- (treat as new span)
            _flush()
            current_tag  = tag[2:] if tag.startswith("I-") else tag
            current_tokens = [word]

    _flush()
    return roles


def extract_roles_hf(clause_text: str) -> list[dict[str, Any]]:
    """
    Extract semantic role frames using yeomtong/srl_bert_model.

    `prediction_formatted` already handles per-verb inference internally
    and returns one frame per detected verb — matching the AllenNLP schema.
    """
    result = _prediction_formatted(clause_text.strip())
    # result = {"words": [...], "verbs": [{"verb": ..., "tags": [...], ...}, ...]}

    words: list[str] = result.get("words", [])
    frames: list[dict[str, Any]] = []

    for verb_info in result.get("verbs", []):
        roles = _decode_bio_spans(words, verb_info["tags"])
        frames.append({
            "predicate": verb_info["verb"],
            "roles":     roles,
        })

    return frames


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
    """Token text + close dependents (det, amod, compound…) in word order."""
    MODIFIER_DEPS = {"compound", "flat", "amod", "det", "nummod", "poss"}
    seen: set[int] = set()
    collected: list[Any] = []

    def _collect(t) -> None:
        if t.i not in seen:
            seen.add(t.i)
            collected.append(t)

    _collect(token)
    for child in token.children:
        if child.dep_ in MODIFIER_DEPS:
            for t in child.subtree:
                _collect(t)

    return " ".join(t.text for t in sorted(collected, key=lambda t: t.i))


def _get_subtree(token) -> str:
    """Full subtree text of a token in sentence order."""
    return " ".join(t.text for t in sorted(token.subtree, key=lambda t: t.i))


def extract_roles_spacy(clause_text: str) -> list[dict[str, Any]]:
    """Extract semantic roles using spaCy dependency parsing (fallback)."""
    nlp  = _get_spacy()
    doc  = nlp(clause_text.strip())
    root = next((t for t in doc if t.dep_ == "ROOT"), None)

    if root is None:
        return [{"predicate": None, "roles": {}}]

    auxiliaries = sorted(
        (c for c in root.children if c.dep_ in ("aux", "auxpass")),
        key=lambda t: t.i,
    )
    predicate = " ".join(t.text for t in auxiliaries + [root]).strip()

    roles: dict[str, str] = {}
    time_parts: list[str] = []

    TIME_PREPS      = {"before", "after", "by", "within", "during", "until", "from", "on"}
    CONDITION_MARKS = {"if", "unless", "provided", "when", "should"}

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
                (gc for gc in child.children if gc.dep_ in ("pobj", "obj")), None
            )
            if pobj:
                prep_text = child.text.lower()
                obj_text  = _get_phrase(pobj)
                if prep_text in ("to", "unto") and "Recipient" not in roles:
                    roles["Recipient"] = obj_text
                elif prep_text in TIME_PREPS:
                    time_parts.append(f"{prep_text} {obj_text}")
        elif dep == "advcl":
            mark = next(
                (gc.text.lower() for gc in child.children if gc.dep_ == "mark"),
                None,
            )
            if mark in CONDITION_MARKS:
                roles["Condition"] = _get_subtree(child)

    if time_parts:
        roles["Time"] = "; ".join(time_parts)

    return [{"predicate": predicate, "roles": roles}]


# ─── Unified interface ───────────────────────────────────────

def extract_semantic_roles(clause_text: str) -> dict[str, Any]:
    """
    Extract semantic roles from a single clause.

    Uses yeomtong/srl_bert_model when available, otherwise falls back
    to spaCy dependency-based extraction.

    Returns:
        {
            "clause":     original clause text,
            "predicate":  main predicate  (frame with most roles),
            "roles":      {role_name: span},
            "all_frames": list of all verb frames,
            "method":     "hf_bert_srl" | "spacy_dependency"
        }
    """
    frames: list[dict[str, Any]] = (
        extract_roles_hf(clause_text)
        if _use_hf
        else extract_roles_spacy(clause_text)
    )

    method = "hf_bert_srl" if _use_hf else "spacy_dependency"

    if not frames:
        return {
            "clause":     clause_text.strip(),
            "predicate":  None,
            "roles":      {},
            "all_frames": [],
            "method":     method,
        }

    best = max(frames, key=lambda f: len(f.get("roles", {})))

    return {
        "clause":     clause_text.strip(),
        "predicate":  best.get("predicate"),
        "roles":      best.get("roles", {}),
        "all_frames": frames,
        "method":     method,
    }


# ─── NER helper ──────────────────────────────────────────────

def _load_ner_results(ner_path: Optional[str]) -> Optional[list[dict]]:
    """Load NER results from JSON; return None if file is absent or invalid."""
    if not ner_path or not os.path.exists(ner_path):
        return None
    try:
        with open(ner_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Could not parse NER file '%s': %s", ner_path, exc)
        return None


# ─── File-level processing ───────────────────────────────────

def process_file(
    input_path: str,
    output_path: str,
    ner_path: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Run SRL on every clause in *input_path* and write results to *output_path*.

    Args:
        input_path:  Plain-text file with one clause per line.
        output_path: Destination JSON file for SRL results.
        ner_path:    Optional path to NER results JSON (Task 2.1 output).

    Returns:
        List of SRL result dicts (one per clause).
    """
    try:
        with open(input_path, "r", encoding="utf-8") as fh:
            clauses = [line.strip() for line in fh if line.strip()]
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
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    method_label = "yeomtong/srl_bert_model" if _use_hf else "spaCy dependency"
    roles_found  = sum(1 for r in results if r["roles"])
    logger.info("Method    : %s", method_label)
    logger.info("Clauses   : %d total, %d with roles", len(clauses), roles_found)
    logger.info("Output    : %s", output_path)

    return results


# ─── Entry point ─────────────────────────────────────────────

if __name__ == "__main__":
    # Expected layout:
    #   <project_root>/
    #       BTL1/output/clauses.txt
    #       BTL2/task2_2_srl.py   ← this file
    #       BTL2/output/ner_results.json
    #       BTL2/output/srl_results.json
    this_file   = Path(__file__).resolve()
    # Go up from srl.py -> src -> BTL2 -> project_root
    project_dir = this_file.parent.parent.parent
    btl2_dir    = project_dir / "BTL2"

    btl1_clauses = project_dir / "BTL1" / "output" / "clauses.txt"
    output_path  = btl2_dir   / "output" / "srl_results.json"
    ner_path     = btl2_dir   / "output" / "ner_results.json"

    # Ensure the input file from BTL1 exists
    if not btl1_clauses.exists():
        logger.error("Input file not found: %s", btl1_clauses)
        logger.error("Please make sure you have run the pipeline for BTL1.")
        sys.exit(1)

    process_file(str(btl1_clauses), str(output_path), str(ner_path))