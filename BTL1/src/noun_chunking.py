"""
Task 1.2: Noun Phrase Chunking
================================
Detect and label noun phrases in each clause using the IOB tagging scheme.

Uses spaCy's built-in noun_chunks to identify noun phrase spans,
then assigns B-NP (beginning), I-NP (inside), or O (outside) tags
to each token.

Input:  output/clauses.txt  (one clause per line from Task 1.1)
Output: output/chunks.txt   (IOB-tagged tokens, one token per line)
"""
import spacy


def load_model():
    """Load the spaCy English model, downloading if necessary."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

def chunk_clause(clause_text, nlp):
    """
    Perform noun phrase chunking on a single clause and return a list of
    (token_text, iob_tag) tuples.

    IOB scheme:
        B-NP  — first token of a noun phrase
        I-NP  — subsequent token inside a noun phrase
        O     — token outside any noun phrase (always used for PUNCT)

    Args:
        clause_text: A single clause string.
        nlp:         Loaded spaCy language model.

    Returns:
        List of (str, str) tuples — (token text, IOB tag).
    """
    doc = nlp(clause_text.strip())
    iob_map = {}  

    for chunk in doc.noun_chunks:
        start, end = chunk.start, chunk.end

        while start < end and doc[start].pos_ in ("PUNCT", "CCONJ"):
            start += 1
        while end > start and doc[end - 1].pos_ in ("PUNCT",):   
            end -= 1

        if start >= end:
            continue

        for i in range(start, end):
            token = doc[i]
            if token.pos_ == "PUNCT":
                continue
            iob_map[i] = "B-NP" if i == start else "I-NP"

    tagged_tokens = []
    for token in doc:
        if token.pos_ == "PUNCT":
            tag = "O"
        else:
            tag = iob_map.get(token.i, "O")
        tagged_tokens.append((token.text, tag))

    return tagged_tokens


def format_iob_output(clause_text, tagged_tokens):
    """
    Format one clause's tagged tokens into the required IOB output block.

    Format:
        # Clause: <original clause text>
        TOKEN<TAB>TAG
        TOKEN<TAB>TAG
        ...
        <blank line>

    Args:
        clause_text:    The original clause string (used in the header comment).
        tagged_tokens:  List of (token_text, iob_tag) tuples.

    Returns:
        A formatted multi-line string ending with a blank line.
    """
    lines = [f"# Clause: {clause_text}"]
    for token_text, tag in tagged_tokens:
        lines.append(f"{token_text}\t{tag}")
    lines.append("")   # blank line separator between clauses
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def process_file(input_path, output_path):
    """
    Read clauses from input_path (one per line), perform IOB noun-phrase
    chunking on each clause, and write the result to output_path.

    Args:
        input_path:  Path to output/clauses.txt
        output_path: Path to output/chunks.txt
    """
    nlp = load_model()

    with open(input_path, "r", encoding="utf-8") as f:
        clauses = [line.strip() for line in f if line.strip()]

    output_blocks = []
    total_np_count = 0

    for clause in clauses:
        tagged_tokens = chunk_clause(clause, nlp)
        block = format_iob_output(clause, tagged_tokens)
        output_blocks.append(block)

        # Count noun phrases (one B-NP = one NP boundary)
        total_np_count += sum(1 for _, tag in tagged_tokens if tag == "B-NP")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_blocks))

    print(
        f"[Noun Phrase Chunking] Processed {len(clauses)} clauses, "
        f"found {total_np_count} noun phrases total.\n"
        f"Output written to: {output_path}"
    )

if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path  = os.path.join(base_dir, "output", "clauses.txt")
    output_path = os.path.join(base_dir, "output", "chunks.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    process_file(input_path, output_path)