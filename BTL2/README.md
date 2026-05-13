# Assignment 2: Information Extraction & Semantic Analysis

## 🎯 Objective
Transform structured clauses into meaningful semantic representations that capture "who does what" and the specific legal intent of each clause.

## 🛠️ Key Tasks

### 1. Domain-Specific NER (Named Entity Recognition)
- **Input**: Preprocessed clauses from BTL1.
- **Output**: `output/ner_results.json`
- **Goal**: Extract critical legal entities: `PARTY`, `MONEY`, `DATE`, `RATE`, `PENALTY`, `LAW`.
- **Model**: Custom-trained models optimized for legal vocabulary.

### 2. Semantic Role Labeling (SRL)
- **Input**: Entities from NER and preprocessed clauses.
- **Output**: `output/srl_results.json`
- **Goal**: Map functional roles (`Agent`, `Predicate`, `Theme`, `Recipient`, `Time`, `Condition`) to understand the dynamics of the agreement.

### 3. Clause Intent Classification
- **Input**: Clauses from BTL1.
- **Output**: `output/intent_classification.txt`
- **Goal**: Classify clauses into four primary legal categories:
  - **Obligation**: What must be done.
  - **Prohibition**: What must not be done.
  - **Right**: What a party is entitled to.
  - **Termination**: Conditions for ending the agreement.

## 🚀 Execution
Run extraction and classification:
```bash
# Standard run (includes model training if required)
python src/main.py 

# Quick inference (skips DistilBERT training)
python src/main.py --no-transformer
```
