import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def run_ingestion():
    print("Starting data ingestion...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clauses_path = os.path.join(base_dir, "data1", "BTL1", "clauses.txt")
    intents_path = os.path.join(base_dir, "data1", "BTL2", "intent_classification.txt")
    
    # Read data
    clauses = []
    intents = []
    
    if os.path.exists(clauses_path):
        with open(clauses_path, "r", encoding="utf-8") as f:
            clauses = [line.strip() for line in f if line.strip()]
    else:
        print(f"Warning: {clauses_path} not found.")
        
    if os.path.exists(intents_path):
        with open(intents_path, "r", encoding="utf-8") as f:
            # Assuming simple line-by-line intent mapping
            intents = [line.strip() for line in f if line.strip()]
    else:
        print(f"Warning: {intents_path} not found.")

    docs = []
    for i, clause in enumerate(clauses):
        intent = intents[i] if i < len(intents) else "Unknown"
        doc = Document(
            page_content=clause,
            metadata={"intent": intent, "source": f"Clause {i+1}"}
        )
        docs.append(doc)

    if not docs:
        print("No documents to ingest!")
        return

    print(f"Loaded {len(docs)} documents. Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    persist_directory = os.path.join(base_dir, "chroma_db")
    
    print("Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Ingestion complete! Database saved to {persist_directory}")

if __name__ == "__main__":
    run_ingestion()
