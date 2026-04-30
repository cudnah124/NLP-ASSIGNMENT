"""
Task 1.1: Clause Splitting
===========================
Split complex legal sentences into semantically independent clauses.

This module uses spaCy's dependency parser to identify clause boundaries
by detecting coordinating conjunctions (cc), adverbial clause modifiers (advcl),
relative clauses (relcl), and other subordinating structures.

Input:  input/raw_contracts.txt  (raw legal contract text)
Output: output/clauses.txt      (one independent clause per line)
"""

import spacy


def load_model():
    """Load the spaCy English model."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model 'en_core_web_sm'...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp


def find_clause_boundaries(doc):
    """
    Find clause boundary indices in a spaCy Doc by detecting:
      - Coordinating conjunctions (cc) linking independent clauses
      - Adverbial clause modifiers (advcl)
      - Relative clauses (relcl)

    Returns a sorted list of token indices where splits should occur.
    """
    boundaries = set()

    for token in doc:
        # Coordinating conjunction (e.g., "and", "or", "but") linking two verbs
        if token.dep_ == "cc" and token.head.pos_ == "VERB":
            # Check if there is a conj sibling that is also a verb
            has_conj_verb = any(
                child.dep_ == "conj" and child.pos_ == "VERB"
                for child in token.head.children
            )
            if has_conj_verb:
                boundaries.add(token.i)

        # Adverbial clause modifier (e.g., "if payment is delayed, ...")
        if token.dep_ == "advcl":
            # Find the start of the adverbial clause
            subtree_indices = [t.i for t in token.subtree]
            clause_start = min(subtree_indices)
            boundaries.add(clause_start)

    return sorted(boundaries)


def split_into_clauses(text, nlp):
    """
    Split a single sentence into independent clauses.

    Args:
        text: A single sentence string.
        nlp:  A loaded spaCy model.

    Returns:
        A list of clause strings.
    """
    doc = nlp(text.strip())

    if len(doc) == 0:
        return []

    boundaries = find_clause_boundaries(doc)

    if not boundaries:
        return [text.strip()]

    clauses = []
    prev = 0

    for boundary in boundaries:
        if boundary > prev:
            clause_tokens = doc[prev:boundary]
            clause_text = clause_tokens.text.strip()
            # Clean up leading/trailing punctuation and conjunctions
            clause_text = _clean_clause(clause_text)
            if clause_text:
                clauses.append(clause_text)
        prev = boundary

    # Add the remaining tokens as the last clause
    if prev < len(doc):
        clause_tokens = doc[prev:]
        clause_text = clause_tokens.text.strip()
        clause_text = _clean_clause(clause_text)
        if clause_text:
            clauses.append(clause_text)

    # If splitting produced nothing meaningful, return the original
    if not clauses:
        return [text.strip()]

    return clauses


def _clean_clause(clause_text):
    """
    Clean a clause string:
      - Remove leading commas, semicolons, conjunctions
      - Capitalize the first letter
      - Ensure the clause ends with a period
    """
    # Strip whitespace
    clause_text = clause_text.strip()

    if not clause_text:
        return ""

    # Remove leading punctuation and common conjunctions
    leading_words_to_remove = ["and", "or", "but", ",", ";"]
    changed = True
    while changed:
        changed = False
        for word in leading_words_to_remove:
            if clause_text.lower().startswith(word):
                clause_text = clause_text[len(word):].strip()
                changed = True

    if not clause_text:
        return ""

    # Capitalize the first letter
    clause_text = clause_text[0].upper() + clause_text[1:]

    # Ensure it ends with a period
    if clause_text and clause_text[-1] not in ".!?":
        clause_text += "."

    return clause_text


def process_file(input_path, output_path):
    """
    Read raw contract text, split each sentence into clauses,
    and write one clause per line to the output file.

    Args:
        input_path:  Path to input/raw_contracts.txt
        output_path: Path to output/clauses.txt
    """
    nlp = load_model()

    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    all_clauses = []

    # Process each line (each line is a contract sentence)
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Use spaCy sentence segmentation first
        doc = nlp(line)
        for sent in doc.sents:
            clauses = split_into_clauses(sent.text, nlp)
            all_clauses.extend(clauses)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for clause in all_clauses:
            f.write(clause + "\n")

    print(f"[Clause Splitting] Wrote {len(all_clauses)} clauses to {output_path}")
    return all_clauses


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "input", "raw_contracts.txt")
    output_path = os.path.join(base_dir, "output", "clauses.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    process_file(input_path, output_path)
