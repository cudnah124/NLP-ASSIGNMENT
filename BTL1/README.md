# Assignment 1: Preprocessing and Syntax Analysis

## Objective
This assignment focuses on building the foundational components for processing legal contract documents. Transform unstructured contract text into syntactically structured representations.

## Tasks
1. **Clause Splitting (1.1)**
   - **Input:** `input/raw_contracts.txt`
   - **Output:** `output/clauses.txt` (One independent clause per line)
   - **Goal:** Split a complex sentence into semantically independent clauses.

2. **Noun Phrase Chunking (1.2)**
   - **Input:** `output/clauses.txt`
   - **Output:** `output/chunks.txt`
   - **Goal:** Detect and label noun phrases in each clause using the IOB tagging scheme (B-NP, I-NP, O).

3. **Dependency Analysis (1.3)**
   - **Input:** `output/clauses.txt`
   - **Output:** `output/dependency.json` (or CoNLL-U format)
   - **Goal:** Perform dependency parsing to determine head-dependent relationships and syntactic roles.
