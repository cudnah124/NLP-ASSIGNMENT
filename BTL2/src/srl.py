"""
Task 2.2: Semantic Role Labeling (SRL)
========================================
Uses yeomtong/srl_bert_model (HuggingFace) for BERT-based SRL.

For each clause, the model is run once per detected verb (predicate-aware
inference). BIO tags produced by the model are decoded and mapped to
human-readable role names.

If the HuggingFace model is unavailable, falls back to spaCy
dependency-based extraction.

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

# ─── Model identifier ────────────────────────────────────────

HF_MODEL_NAME = "yeomtong/srl_bert_model"

# ─── Attempt to load HuggingFace SRL model ───────────────────

_hf_tokenizer = None
_hf_model = None
_use_hf = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    logger.info("Loading tokenizer from '%s' …", HF_MODEL_NAME)
    _hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    logger.info("Loading model from '%s' …", HF_MODEL_NAME)
    _hf_model = AutoModelForTokenClassification.from_pretrained(HF_MODEL_NAME)
    _hf_model.eval()

    _use_hf = True
    logger.info("HuggingFace SRL model loaded successfully.")
except Exception as exc:
    logger.warning("HuggingFace model unavailable (%s). Falling back to spaCy.", exc)


# ─── PropBank → readable role name mapping ───────────────────

# Labels come dynamically from model.config.id2label at runtime;
# this dict maps the raw PropBank tag (without B-/I- prefix) to a
# human-readable name used in the output JSON.
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
    # "V" is intentionally absent: the verb is stored in `predicate`, not roles
}


# ─── HuggingFace BERT-based SRL ──────────────────────────────

def _get_words(text: str) -> list[str]:
    """Whitespace-tokenize text into words (pre-tokenization)."""
    return text.strip().split()


def _predict_tags_for_verb(
    words: list[str],
    verb_index: int,
) -> list[str]:
    """
    Run one forward pass of the SRL model for a single predicate.

    Many BERT SRL models (e.g. the AllenNLP family that yeomtong is based on)
    expect a predicate *indicator* — a binary sequence marking which token is
    the verb — passed as `token_type_ids`.  We encode this via token_type_ids:
      0 for ordinary tokens, 1 for the predicate token.

    Returns a list of raw BIO tags aligned to `words`.
    """
    import torch

    # Tokenize with word-level offset tracking
    encoding = _hf_tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    # Build predicate indicator aligned to subword tokens
    word_ids: list[Optional[int]] = encoding.word_ids(batch_index=0)
    token_type_ids = torch.zeros_like(encoding["input_ids"])
    for pos, wid in enumerate(word_ids):
        if wid == verb_index:
            token_type_ids[0, pos] = 1
    encoding["token_type_ids"] = token_type_ids

    with torch.no_grad():
        outputs = _hf_model(**encoding)

    # Decode logits → label ids → label strings (keep only first subword per word)
    label_ids: list[int] = outputs.logits.argmax(dim=-1)[0].tolist()
    id2label: dict[int, str] = _hf_model.config.id2label

    # Align subword predictions back to the original word list
    word_tags: list[str] = []
    seen_word: set[int] = set()
    for pos, wid in enumerate(word_ids):
        if wid is None or wid in seen_word:
            continue
        seen_word.add(wid)
        word_tags.append(id2label[label_ids[pos]])

    return word_tags


def _decode_bio_spans(
    words: list[str],
    tags: list[str],
) -> dict[str, str]:
    """
    Convert a BIO tag sequence into a {role_name: span_text} dict.

    Handles both "B-LABEL / I-LABEL" and flat "LABEL" (no B/I prefix).
    The "V" label (predicate) is skipped — it is stored separately.
    """
    roles: dict[str, str] = {}
    current_tag: Optional[str] = None
    current_tokens: list[str] = []

    def _flush() -> None:
        if current_tag and current_tokens and current_tag != "V":
            role_name = ROLE_MAP.get(current_tag, current_tag)
            roles[role_name] = " ".join(current_tokens)

    for word, tag in zip(words, tags):
        if tag.startswith("B-") or tag.startswith("I-"):
            prefix, raw_tag = tag[:2], tag[2:]
        else:
            prefix, raw_tag = "", tag

        if tag == "O":
            _flush()
            current_tag, current_tokens = None, []
        elif prefix == "B-":
            _flush()
            current_tag, current_tokens = raw_tag, [word]
        elif prefix == "I-" and current_tag == raw_tag:
            current_tokens.append(word)
        else:
            # Flat label (no BIO prefix) or unexpected I- without matching B-
            _flush()
            current_tag, current_tokens = raw_tag, [word]

    _flush()
    return roles


def _detect_verb_indices(words: list[str]) -> list[int]:
    """
    Return indices of verb tokens using spaCy POS tagging.
    Falls back to a simple heuristic if spaCy is unavailable.
    """
    try:
        nlp = _get_spacy()
        doc = nlp(" ".join(words))
        return [token.i for token in doc if token.pos_ == "VERB"]
    except Exception:
        COMMON_AUX = {
            "is", "are", "was", "were", "be", "been", "being",
            "has", "have", "had", "do", "does", "did",
            "will", "would", "shall", "should", "may",
            "might", "must", "can", "could",
        }
        return [
            i for i, w in enumerate(words)
            if w.lower() not in COMMON_AUX and len(w) > 2
        ]


def extract_roles_hf(clause_text: str) -> list[dict[str, Any]]:
    """
    Extract semantic role frames using yeomtong/srl_bert_model.

    Runs one inference pass per detected verb and returns one frame per verb.
    """
    words = _get_words(clause_text)
    if not words:
        return []

    verb_indices = _detect_verb_indices(words)
    if not verb_indices:
        # No verbs found — attempt a single pass on index 0
        verb_indices = [0]

    frames: list[dict[str, Any]] = []
    for v_idx in verb_indices:
        try:
            tags  = _predict_tags_for_verb(words, v_idx)
            roles = _decode_bio_spans(words, tags)
            frames.append({"predicate": words[v_idx], "roles": roles})
        except Exception as exc:
            logger.debug(
                "Skipping verb '%s' at index %d: %s", words[v_idx], v_idx, exc
            )

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
    """
    Return the text of a token together with its close dependents
    (determiners, modifiers, compounds), preserving original word order.
    """
    MODIFIER_DEPS = {"compound", "flat", "amod", "det", "nummod", "poss"}
    seen: set[int] = set()
    collected: list[Any] = []

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

    Uses yeomtong/srl_bert_model when available, otherwise falls back to
    spaCy dependency-based extraction.

    Returns:
        {
            "clause":     original clause text,
            "predicate":  main predicate string  (best frame),
            "roles":      {role_name: span, …}   (best frame),
            "all_frames": [{predicate, roles}, …],
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
    """Load NER results from JSON, returning None if unavailable."""
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

    method_label = "HuggingFace BERT SRL" if _use_hf else "spaCy dependency"
    roles_found  = sum(1 for r in results if r["roles"])
    logger.info(
        "Method    : %s (%s)",
        method_label,
        HF_MODEL_NAME if _use_hf else "en_core_web_sm",
    )
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
    btl2_dir    = this_file.parent
    project_dir = btl2_dir.parent

    btl1_clauses = project_dir / "BTL1" / "output" / "clauses.txt"
    test_clauses = btl2_dir / "input" / "clauses.txt"  # For testing without BTL1 output
    output_path  = btl2_dir   / "output" / "srl_results.json"
    ner_path     = btl2_dir   / "output" / "ner_results.json"

    process_file(str(test_clauses), str(output_path), str(ner_path))