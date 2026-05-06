"""
Task 2.2: Semantic Role Labeling (SRL).

HF-only inference using `yeomtong/srl_bert_model`.
Input: clauses.txt (from BTL1) + optional NER output.
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

HF_REPO_ID   = "yeomtong/srl_bert_model"
HF_CKPT_FILE = "best_srl_Sep_29.ckpt"
BERT_NAME    = "bert-base-cased"

_prediction_formatted = None

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
    logger.info("yeomtong/srl_bert_model loaded successfully.")

except Exception as exc:
    raise RuntimeError(
        "Failed to load SRL model from HuggingFace Hub. "
        "BTL2 SRL requires HF-only inference; non-HF fallback is disabled.\n"
        f"Original error: {exc}"
    ) from exc


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
            _flush()
            current_tag  = tag[2:] if tag.startswith("I-") else tag
            current_tokens = [word]

    _flush()
    return roles


def extract_roles_hf(clause_text: str) -> list[dict[str, Any]]:
    """
    Extract semantic role frames using yeomtong/srl_bert_model.

    `prediction_formatted` already handles per-verb inference internally
    and returns one frame per detected verb.
    """
    result = _prediction_formatted(clause_text.strip())

    words: list[str] = result.get("words", [])
    frames: list[dict[str, Any]] = []

    for verb_info in result.get("verbs", []):
        roles = _decode_bio_spans(words, verb_info["tags"])
        frames.append({
            "predicate": verb_info["verb"],
            "roles":     roles,
        })

    return frames


def extract_semantic_roles(clause_text: str) -> dict[str, Any]:
    """
    Extract semantic roles from a single clause.

    Uses yeomtong/srl_bert_model (HF-only SRL).

    Returns:
        {
            "clause":     original clause text,
            "predicate":  main predicate  (frame with most roles),
            "roles":      {role_name: span},
            "all_frames": list of all verb frames,
            "method":     "hf_bert_srl"
        }
    """
    frames: list[dict[str, Any]] = extract_roles_hf(clause_text)
    method = "hf_bert_srl"

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

    roles_found  = sum(1 for r in results if r["roles"])
    logger.info("Method    : yeomtong/srl_bert_model (HF only)")
    logger.info("Clauses   : %d total, %d with roles", len(clauses), roles_found)
    logger.info("Output    : %s", output_path)

    return results


if __name__ == "__main__":
    this_file   = Path(__file__).resolve()
    project_dir = this_file.parent.parent.parent
    btl2_dir    = project_dir / "BTL2"

    btl1_clauses = btl2_dir / "input" / "clauses.txt"  
    output_path  = btl2_dir   / "output" / "srl_results.json"
    ner_path     = btl2_dir   / "output" / "ner_results.json"

    process_file(str(btl1_clauses), str(output_path), str(ner_path))