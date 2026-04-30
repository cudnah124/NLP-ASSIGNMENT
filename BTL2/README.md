# Assignment 2: Information Extraction and Semantic Analysis

## Objective
Transform syntactically structured clauses into meaningful representations that capture "who does what" and the legal function of each clause within the contract.

## Tasks
1. **Custom Named Entity Recognition (NER) (2.1)**
   - **Input:** Clauses from Assignment 1 and Annotated training dataset.
   - **Output:** `output/ner_results.json`
   - **Goal:** Design and train/fine-tune a domain-specific NER model for entities like PARTY, MONEY, DATE, RATE, PENALTY, LAW.

2. **Semantic Role Labeling (SRL) (2.2)**
   - **Input:** Clauses from Assignment 1 and Named entities from 2.1.
   - **Output:** `output/srl_results.json`
   - **Goal:** Assign semantic roles (Agent, Predicate, Theme, Recipient, Time, Condition) to entities with respect to the main predicate.

3. **Clause Intent Classification (2.3)**
   - **Input:** Clauses from Assignment 1.
   - **Output:** `output/intent_classification.txt`
   - **Goal:** Classify each clause into Obligation, Prohibition, Right, or Termination Condition.
