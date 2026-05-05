import re
import spacy

def load_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp


def find_subject_node(verb_token):
    for child in verb_token.children:
        if child.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass"):
            return child
    if verb_token.dep_ == "conj":
        return find_subject_node(verb_token.head)
    return None


def get_core_np(token):
    SKIP_DEPS = {"relcl", "prep", "appos", "acl", "advcl", "cc", "conj"}
    core_tokens = []
    for t in token.subtree:
        if t == token:
            core_tokens.append(t)
        elif t.head == token and t.dep_ in ("det", "compound", "amod", "poss", "nummod"):
            core_tokens.append(t)
        elif t.head != token and t.dep_ not in SKIP_DEPS:
            if t.dep_ in ("det", "compound", "amod", "poss", "nummod"):
                core_tokens.append(t)
    core_tokens.sort(key=lambda x: x.i)
    return " ".join(t.text for t in core_tokens)


def get_aux_phrase(verb_token):
    auxs = [c for c in verb_token.children if c.dep_ in ("aux", "auxpass", "neg")]
    if not auxs and verb_token.dep_ == "conj":
        return get_aux_phrase(verb_token.head)
    auxs.sort(key=lambda x: x.i)
    return " ".join(t.text for t in auxs)


SUBORD_CONJ = re.compile(
    r"^(when|while|although|though|even\s+though|since|because|as\s+(?:long\s+as)?"
    r"|unless|after|before|once|until|if|in\s+order\s+to|so\s+that|provided\s+that"
    r"|notwithstanding|except\s+(?:where|when|that)?)\b",
    re.IGNORECASE,
)


def is_part_of_list(cc_token):
    head = cc_token.head

    if head.pos_ in ("NOUN", "PROPN", "ADJ", "NUM"):
        return True

    all_conjs = [t for t in head.subtree if t.dep_ == "conj" and t != head]
    if len(all_conjs) >= 2:
        return True

    for offset in (1, 2):
        idx = cc_token.i - offset
        if idx >= 0 and cc_token.doc[idx].text == ",":
            sibling_conjs = [
                c for c in head.children
                if c.dep_ == "conj" and c.i < cc_token.i
            ]
            if sibling_conjs:
                return True

    return False


def contains_provided_that(text):
    """Return True if text contains a 'provided that' conditional chain."""
    return bool(re.search(r"\bprovided\s+that\b", text, re.IGNORECASE))


def _has_independent_clause(text, nlp):
    """
    Return True if the text contains a complete independent clause —
    i.e. a ROOT verb that has its own nominal subject.

    This is used to distinguish between:
      • A genuine dependent fragment:
            "Although he tried." — ROOT=tried, nsubj=he BUT the whole
            clause is purely subordinate (no second main verb).
      • A sentence that merely *begins* with a subordinate clause
        but also contains a full main clause:
            "If payment is delayed, a penalty shall apply."
            ROOT=apply, nsubj=penalty → independent clause present.

    Strategy: find the ROOT token; if it sits *inside* a subordinate
    advcl or has a direct nsubj/nsubjpass child, classify accordingly.
    A ROOT with an nsubj child that is NOT itself governed by the
    leading subordinate conjunction means the sentence is self-contained.
    """
    doc = nlp(text.strip())
    for token in doc:
        if token.dep_ == "ROOT":
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    return True
    return False


def split_into_clauses(text, nlp):
    """
    Split one sentence into semantically independent clauses.
    Returns a list of cleaned clause strings.
    """
    doc = nlp(text.strip())
    if len(doc) == 0:
        return []

    split_points = []

    for token in doc:
        if token.text == ";" and token.i < len(doc) - 1:
            split_points.append(token.i + 1)
            continue

        if token.dep_ == "cc" and token.text.lower() in ("and", "or", "but"):
            if is_part_of_list(token):
                continue

            head = token.head
            conjuncts = [
                c for c in head.children
                if c.dep_ == "conj" and c.i > token.i
                and c.pos_ in ("VERB", "AUX")
            ]
            for conj in conjuncts:
                has_own_subj = any(
                    c.dep_ in ("nsubj", "nsubjpass") for c in conj.children
                )
                subtree_len = len(list(conj.subtree))

                # Also split when the conjunct's subtree contains an advcl
                # introduced by a subordinate conjunction (e.g. "if …, V")
                # — these form a complete conditional main clause.
                has_advcl = any(
                    c.dep_ == "advcl" for c in conj.subtree
                )

                if has_own_subj or (has_advcl and subtree_len > 6):
                    split_points.append(token.i)
                    break
                if subtree_len > 8:
                    split_points.append(token.i)
                    break

    raw_segments = []
    prev_idx = 0
    sorted_splits = sorted(set(split_points))

    for split_idx in sorted_splits + [len(doc)]:
        if split_idx > prev_idx:
            raw_segments.append((prev_idx, split_idx))
        prev_idx = (
            split_idx + 1
            if split_idx < len(doc) and doc[split_idx - 1].text in (";",)
            else split_idx
        )

    clauses = []
    current_subject_text = None
    current_aux_text = None

    for seg_start, seg_end in raw_segments:
        segment_tokens = doc[seg_start:seg_end]
        if not segment_tokens:
            continue

        seg_text = segment_tokens.text.strip()
        seg_text = re.sub(r"^(and|or|but)[,\s]+", "", seg_text, flags=re.IGNORECASE).strip()

        lexical_verbs = [
            t for t in segment_tokens
            if t.pos_ == "VERB" and t.dep_ not in ("xcomp", "ccomp", "advcl", "relcl")
        ]
        aux_verbs = [
            t for t in segment_tokens
            if t.pos_ == "AUX" and t.dep_ not in ("xcomp", "ccomp", "advcl", "relcl")
        ]
        segment_verbs = lexical_verbs if lexical_verbs else aux_verbs

        needs_propagation = False
        if segment_verbs:
            main_verb = segment_verbs[0]
            subj_node = find_subject_node(main_verb)

            if subj_node is not None:
                subj_in_segment = seg_start <= subj_node.i < seg_end
                if subj_in_segment:
                    current_subject_text = get_core_np(subj_node)
                    current_aux_text = get_aux_phrase(main_verb)
                else:
                    needs_propagation = True
            else:
                needs_propagation = True

        # Reconstruct the clause string
        if needs_propagation and current_subject_text:
            aux_part = (" " + current_aux_text) if current_aux_text else ""
            seg_text = current_subject_text + aux_part + " " + seg_text

        cleaned = _clean_clause(seg_text)
        if cleaned:
            clauses.append(cleaned)

    merged = _merge_dependent_fragments(clauses, nlp)
    return merged if merged else [text.strip()]


def _merge_dependent_fragments(clauses, nlp=None):
    """
    Merge a clause that is a bare dependent fragment (e.g. "Before signing.")
    back onto the previous clause.

    A clause that *starts* with a subordinate conjunction is only merged
    if it does NOT contain its own independent clause (root verb + subject).
    This prevents sentences like "If payment is delayed, a penalty shall apply."
    from being incorrectly merged back after the split on "and".
    """
    if not clauses:
        return clauses

    result = []
    i = 0
    while i < len(clauses):
        clause = clauses[i]
        stripped = clause.rstrip(".")

        if SUBORD_CONJ.match(stripped) and result:
            # Only merge if the segment lacks its own independent clause.
            # If it HAS a root + subject it is self-contained and stays separate.
            if nlp is not None and _has_independent_clause(clause, nlp):
                result.append(clause)
            else:
                prev = result[-1].rstrip(".")
                result[-1] = prev + ", " + clause[0].lower() + clause[1:]
        else:
            result.append(clause)
        i += 1
    return result


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_clause(text):
    text = text.strip()
    if not text:
        return ""
    leading_cc = re.compile(r"^(and|or|but)\s*[,;]?\s*", re.IGNORECASE)
    while True:
        m = leading_cc.match(text)
        if not m:
            break
        text = text[m.end():].strip()

    text = text.strip(",;: ")
    if not text:
        return ""
    text = re.sub(r"^\s*\([a-z0-9]+\)\s*", "", text, flags=re.IGNORECASE).strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."

    return text


def process_file(input_path, output_path):
    nlp = load_model()

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    all_clauses = []

    for line in lines:
        line = line.strip()

        if not line:
            continue
        if set(line) <= set("=-"):
            continue
        if re.match(r"^\[EC-\d+\]", line):
            continue
        if line.startswith("=") or line.startswith("-"):
            continue

        doc = nlp(line)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            clauses = split_into_clauses(sent_text, nlp)
            all_clauses.extend(clauses)

    with open(output_path, "w", encoding="utf-8") as f:
        for clause in all_clauses:
            f.write(clause + "\n")


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    process_file(
        os.path.join(base_dir, "input", "raw_contracts.txt"),
        os.path.join(base_dir, "output", "clauses.txt"),
    )