"""
Task 1.2: Noun Phrase Chunking
================================
Detect and label noun phrases in each clause using the IOB tagging scheme.

Uses spaCy's built-in noun_chunks to identify noun phrase spans,
then assigns B-NP (beginning), I-NP (inside), or O (outside) tags
to each token.

Input:  output/clauses.txt  (one clause per line from Task 1.1)
Output: output/chunks.txt   (IOB-tagged tokens)
"""

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


def chunk_clause(clause_text, nlp):
    """
    Perform noun phrase chunking on a single clause.

    Args:
        clause_text: A single clause string.
        nlp:         A loaded spaCy model.

    Returns:
        A list of (token_text, iob_tag) tuples.

    Example:
        Input:  "Party B shall pay the rental amount."
        Output: [("Party", "B-NP"), ("B", "I-NP"), ("shall", "O"),
                 ("pay", "O"), ("the", "B-NP"), ("rental", "I-NP"),
                 ("amount", "I-NP"), (".", "O")]
    """
    doc = nlp(clause_text.strip())

    # Build a set of (start, end) spans for noun chunks
    np_spans = []
    for chunk in doc.noun_chunks:
        np_spans.append((chunk.start, chunk.end))

    # Assign IOB tags
    tagged_tokens = []
    for token in doc:
        tag = "O"
        for start, end in np_spans:
            if token.i == start:
                tag = "B-NP"
                break
            elif start < token.i < end:
                tag = "I-NP"
                break
        tagged_tokens.append((token.text, tag))

    return tagged_tokens


def format_iob_output(tagged_tokens):
    """
    Format tagged tokens into the IOB output format.
    Each line contains: TOKEN\tTAG

    Args:
        tagged_tokens: List of (token_text, iob_tag) tuples.

    Returns:
        A formatted string with one token per line.
    """
    lines = []
    for token_text, tag in tagged_tokens:
        lines.append(f"{token_text}\t{tag}")
    return "\n".join(lines)


def process_file(input_path, output_path):
    """
    Read clauses from file, perform noun phrase chunking,
    and write IOB-tagged output.

    Args:
        input_path:  Path to output/clauses.txt
        output_path: Path to output/chunks.txt
    """
    nlp = load_model()

    with open(input_path, "r", encoding="utf-8") as f:
        clauses = [line.strip() for line in f if line.strip()]

    all_output = []
    total_np_count = 0

    for clause in clauses:
        tagged_tokens = chunk_clause(clause, nlp)
        formatted = format_iob_output(tagged_tokens)
        all_output.append(f"# Clause: {clause}")
        all_output.append(formatted)
        all_output.append("")  # Blank line separator between clauses

        # Count noun phrases for reporting
        np_count = sum(1 for _, tag in tagged_tokens if tag == "B-NP")
        total_np_count += np_count

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_output))

    print(f"[Noun Phrase Chunking] Processed {len(clauses)} clauses, "
          f"found {total_np_count} noun phrases. Output: {output_path}")
    return all_output


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "output", "clauses.txt")
    output_path = os.path.join(base_dir, "output", "chunks.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    process_file(input_path, output_path)
