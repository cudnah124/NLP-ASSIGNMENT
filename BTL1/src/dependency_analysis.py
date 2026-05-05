"""
Task 1.3: Dependency Analysis
===============================
Perform dependency parsing on each clause to determine head-dependent
relationships and syntactic roles.

Uses spaCy's dependency parser to extract, for each token:
  - Token text
  - Head word
  - Dependency relation (e.g., root, nsubj, obj, advcl)

Additionally, a simplified tree is produced that highlights the key
syntactic roles: root (main verb), nsubj (nominal subject), and obj
(direct object) — matching the required output format:

    Example for "Party B shall pay the rental amount.":
        pay     -> root
        Party_B -> nsubj (head: pay)
        amount  -> obj   (head: pay)

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
          - head:     The head word
          - dep_rel:  Dependency relation label

    Example output for "Party B shall pay the rental amount.":
    [
        {"token": "Party",  "head": "pay",    "dep_rel": "nsubj"},
        {"token": "B",      "head": "Party",  "dep_rel": "flat"},
        {"token": "shall",  "head": "pay",    "dep_rel": "aux"},
        {"token": "pay",    "head": "pay",    "dep_rel": "ROOT"},
        {"token": "the",    "head": "amount", "dep_rel": "det"},
        {"token": "rental", "head": "amount", "dep_rel": "amod"},
        {"token": "amount", "head": "pay",    "dep_rel": "obj"},
        {"token": ".",      "head": "pay",    "dep_rel": "punct"}
    ]
    """
    doc = nlp(clause_text.strip())

    tokens_info = []
    for token in doc:
        tokens_info.append({
            "token": token.text,
            "head": token.head.text,
            "dep_rel": token.dep_
        })

    return tokens_info


def _collect_span(token):
    """
    Build a compact label for a token by joining it with proper-noun
    parts only (dep ``flat`` or ``compound`` **and** pos ``PROPN``).

    This correctly produces:
      • "Party_B"  — "B" is PROPN + flat/compound → joined
      • "amount"   — "rental" is NOUN compound, NOT PROPN → excluded,
                     so only the head noun "amount" is returned

    Common-noun compounds (e.g. "rental" in "rental amount") and all
    modifiers (det, amod, poss …) are excluded so the label stays
    concise and matches the expected simplified-tree output:

        Party_B -> nsubj (head: pay)
        amount  -> obj   (head: pay)
    """
    JOIN_DEPS = {"flat", "compound"}

    core_tokens = [token]
    for t in token.subtree:
        if (
            t != token
            and t.dep_ in JOIN_DEPS
            and t.pos_ == "PROPN"   # only join proper-noun parts
            and not t.is_punct
        ):
            core_tokens.append(t)

    core_tokens.sort(key=lambda t: t.i)
    return "_".join(t.text for t in core_tokens)


def build_dependency_tree(clause_text, nlp):
    """
    Build a dependency tree representation focused on key syntactic roles.

    The simplified_tree highlights only the most important relations:
      - root  : the main verb of the clause
      - nsubj : nominal subject (e.g. "Party_B")
      - obj   : direct object  (e.g. "amount")

    Returns a dictionary with:
      - clause:          The original clause text
      - root:            The root token text
      - tokens:          Full token-level dependency information (all tokens)
      - simplified_tree: List of strings showing root / nsubj / obj relations

    Args:
        clause_text: A single clause string.
        nlp:         A loaded spaCy model.

    Example simplified_tree for "Party B shall pay the rental amount.":
        [
            "pay -> root",
            "Party_B -> nsubj (head: pay)",
            "amount -> obj (head: pay)"
        ]
    """
    doc = nlp(clause_text.strip())
    tokens_info = parse_dependencies(clause_text, nlp)

    root_token = None
    simplified_tree = []

    # First pass: find root and emit it first
    for token in doc:
        if token.dep_ == "ROOT":
            root_token = token.text
            simplified_tree.append(f"{token.text} -> root")
            break

    # Second pass: emit nsubj and obj with their compact span labels
    KEY_DEPS = {"nsubj", "nsubjpass", "obj", "dobj", "iobj"}
    for token in doc:
        dep = token.dep_.lower()
        # Normalise legacy 'dobj' -> 'obj' for display
        display_dep = "obj" if dep == "dobj" else dep

        if dep in KEY_DEPS:
            span_label = _collect_span(token)
            simplified_tree.append(
                f"{span_label} -> {display_dep} (head: {token.head.text})"
            )

    return {
        "clause": clause_text.strip(),
        "root": root_token,
        "tokens": tokens_info,
        "simplified_tree": simplified_tree,
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
    print("Simplified Tree:")
    for line in dep_info["simplified_tree"]:
        print(f"  {line}")
    print("All Tokens:")
    for t in dep_info["tokens"]:
        print(f"  {t['token']:15s} head={t['head']:15s} dep={t['dep_rel']}")


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "output", "clauses.txt")
    output_path = os.path.join(base_dir, "output", "dependency.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    process_file(input_path, output_path)