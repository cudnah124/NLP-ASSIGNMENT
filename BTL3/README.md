# Assignment 3: Contract Question Answering Application

## Objective
Build an interactive contract question answering system based on structured outputs generated in Assignments 1 and 2.

## Tasks
Create a user-facing application capable of answering questions about the contract in natural language. Choose one of the following approaches:

**Option 1: Rule-Based Query System**
- Design a deterministic query engine operating on structured JSON outputs.
- Map user queries to structured representations (Agent, Predicate, Theme, Time, Intent).

**Option 2: Retrieval-Augmented Generation (RAG) Chatbot**
- Integrate a vector database, embedding model, and an LLM.
- Retrieve top-k relevant clauses and generate an answer with explicit citations.
