# Assignment 1: Syntax Analysis & Preprocessing

## 🎯 Objective
This component builds the foundational layer for processing legal contracts by transforming unstructured text into syntactically structured representations.

## 🛠️ Key Tasks

### 1. Clause Splitting
- **Input**: `input/raw_contracts.txt`
- **Output**: `output/clauses.txt`
- **Goal**: Decompose complex sentences into semantically independent clauses to simplify downstream analysis.

### 2. Noun Phrase (NP) Chunking
- **Input**: `output/clauses.txt`
- **Output**: `output/chunks.txt`
- **Goal**: Identify and label noun phrases using the **IOB tagging scheme** (B-NP, I-NP, O).

### 3. Dependency Parsing
- **Input**: `output/clauses.txt`
- **Output**: `output/dependency.json`
- **Goal**: Map syntactic relationships (head-dependent) and grammatical roles (nsubj, dobj, etc.) to understand sentence structure.

## 🚀 Execution
Run the preprocessing pipeline:
```bash
python src/main.py
```
