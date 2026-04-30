"""
Combines retriever.py and llm_generator.py into a full RAG chain.
"""

from retriever import retrieve_clauses
from llm_generator import generate_answer

def ask_contract(query):
    # 1. Retrieve context
    # context = retrieve_clauses(query)
    # 2. Generate answer
    # answer = generate_answer(query, context)
    # return answer
    pass
