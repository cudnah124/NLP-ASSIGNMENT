"""
Task 1.3: Dependency Analysis
===============================
Perform dependency parsing on each clause to determine head-dependent
relationships and syntactic roles.

Uses spaCy's dependency parser to extract:
  - Token text
  - Part-of-speech tag
  - Head word
  - Dependency relation (e.g., root, nsubj, obj, advcl)

Input:  output/clauses.txt       (one clause per line from Task 1.1)
Output: output/dependency.json   (structured dependency information)
"""

import json
import spacy


def load_model():
    """Load the spaCy English model."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp


def parse_dependencies(clause_text, nlp):
    """
    Parse a single clause and extract dependency information for each token.

    Args:
        clause_text: A single clause string.
        nlp:         A loaded spaCy model.

    Returns:
        A list of dictionaries, one per token, containing:
          - token:    The token text
          - pos:      Part-of-speech tag
          - head:     The head word
          - dep_rel:  Dependency relation label

    Example output for "Party B shall pay the rental amount.":
    [
        {"token": "Party", "pos": "PROPN", "head": "pay",    "dep_rel": "nsubj"},
        {"token": "B",     "pos": "PROPN", "head": "Party",  "dep_rel": "flat"},
        {"token": "shall", "pos": "AUX",   "head": "pay",    "dep_rel": "aux"},
        {"token": "pay",   "pos": "VERB",  "head": "pay",    "dep_rel": "ROOT"},
        {"token": "the",   "pos": "DET",   "head": "amount", "dep_rel": "det"},
        {"token": "rental","pos": "ADJ",   "head": "amount", "dep_rel": "amod"},
        {"token": "amount","pos": "NOUN",  "head": "pay",    "dep_rel": "dobj"},
        {"token": ".",     "pos": "PUNCT", "head": "pay",    "dep_rel": "punct"}
    ]
    """
    doc = nlp(clause_text.strip())

    tokens_info = []
    for token in doc:
        tokens_info.append({
            "token": token.text,
            "pos": token.pos_,
            "head": token.head.text,
            "dep_rel": token.dep_
        })

    return tokens_info


def build_dependency_tree(clause_text, nlp):
    """
    Build a simplified dependency tree representation.

    Returns a dictionary with:
      - clause: The original clause text
      - root:   The root token
      - tokens: Full token-level dependency information
      - tree:   A human-readable tree string

    Args:
        clause_text: A single clause string.
        nlp:         A loaded spaCy model.
    """
    doc = nlp(clause_text.strip())

    tokens_info = parse_dependencies(clause_text, nlp)

    # Find the root
    root_token = None
    for token in doc:
        if token.dep_ == "ROOT":
            root_token = token.text
            break

    # Build a readable tree representation
    tree_lines = []
    for token in doc:
        if token.dep_ == "ROOT":
            tree_lines.append(f"{token.text} -> root")
        else:
            tree_lines.append(f"{token.text} -> {token.dep_} (head: {token.head.text})")

    return {
        "clause": clause_text.strip(),
        "root": root_token,
        "tokens": tokens_info,
        "tree": tree_lines
    }


def process_file(input_path, output_path):
    """
    Read clauses from file, perform dependency parsing,
    and write structured output to JSON.

    Args:
        input_path:  Path to output/clauses.txt
        output_path: Path to output/dependency.json
    """
    nlp = load_model()

    with open(input_path, "r", encoding="utf-8") as f:
        clauses = [line.strip() for line in f if line.strip()]

    results = []

    for clause in clauses:
        dep_info = build_dependency_tree(clause, nlp)
        results.append(dep_info)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[Dependency Analysis] Parsed {len(clauses)} clauses. Output: {output_path}")
    return results


def print_tree(clause_text, nlp):
    """Pretty-print the dependency tree for a single clause (for debugging)."""
    dep_info = build_dependency_tree(clause_text, nlp)
    print(f"\nClause: {dep_info['clause']}")
    print(f"Root:   {dep_info['root']}")
    print("Tree:")
    for line in dep_info["tree"]:
        print(f"  {line}")


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "output", "clauses.txt")
    output_path = os.path.join(base_dir, "output", "dependency.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    process_file(input_path, output_path)
