import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

HF_REPO_ID   = "yeomtong/srl_bert_model"
HF_CKPT_FILE = "best_srl_Sep_29.ckpt"
BERT_NAME    = "bert-base-cased"

_prediction_formatted = None

try:
    from huggingface_hub import hf_hub_download, snapshot_download

    ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_CKPT_FILE)
    repo_dir = snapshot_download(HF_REPO_ID)
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

    from predictor import srl_init
    from visualizer import prediction_formatted

    srl_init(ckpt_path, bert_name=BERT_NAME)

    _prediction_formatted = prediction_formatted

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

ROLE_PREP_STRIP: dict[str, list[str]] = {
    "Recipient": ["to", "for"],
    "Beneficiary": ["for"],
    "Purpose": ["to", "for"],
    "Condition": ["if", "unless", "when", "provided"],
    "Location": ["in", "at", "on", "into", "inside", "within", "from", "to"],
    "Direction": ["to", "toward", "towards", "into", "onto", "from"],
    "Source": ["from"],
}


def _normalize_role_span(role_name: str, span_text: str) -> str:
    text = span_text.strip()

    for prep in ROLE_PREP_STRIP.get(role_name, []):
        lower_text = text.lower()
        if lower_text.startswith(prep + " "):
            text = text[len(prep) + 1 :].lstrip()
            break

    text = text.strip(" \t\n\r,.;:!?")
    return text


def _decode_bio_spans(words: list[str], tags: list[str]) -> dict[str, str]:
    roles: dict[str, str] = {}
    current_tag: Optional[str] = None
    current_tokens: list[str] = []

    def _flush() -> None:
        if current_tag and current_tokens and current_tag != "V":
            role_name = ROLE_MAP.get(current_tag, current_tag)
            span_text = " ".join(current_tokens)
            roles[role_name] = _normalize_role_span(role_name, span_text)

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
    if not ner_path or not os.path.exists(ner_path):
        return None
    try:
        with open(ner_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, ValueError):
        return None


def process_file(
    input_path: str,
    output_path: str,
    ner_path: Optional[str] = None,
) -> list[dict[str, Any]]:
    try:
        with open(input_path, "r", encoding="utf-8") as fh:
            clauses = [line.strip() for line in fh if line.strip()]
    except OSError:
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

    return results


if __name__ == "__main__":
    this_file   = Path(__file__).resolve()
    project_dir = this_file.parent.parent.parent
    btl2_dir    = project_dir / "BTL2"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    btl1_clauses = os.path.join(base_dir, "..", "BTL1", "output", "clauses.txt")
    output_path  = btl2_dir   / "output" / "srl_results.json"
    ner_path     = btl2_dir   / "output" / "ner_results.json"

    process_file(str(btl1_clauses), str(output_path), str(ner_path))