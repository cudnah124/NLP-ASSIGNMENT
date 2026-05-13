# Assignment 3: Legal Contract QA System (RAG)

## 🎯 Objective
Build an intelligent, interactive Question Answering system that allows users to query legal contracts in natural language using advanced NLP techniques.

## 🛠️ Features

### Retrieval-Augmented Generation (RAG)
The system leverages a modern RAG architecture to ensure accuracy and groundedness:
- **Vector Store**: ChromaDB for efficient semantic search.
- **Embeddings**: Google Generative AI embeddings.
- **LLM**: Google Gemini API for high-quality answer generation.
- **Citations**: Every answer includes the specific clauses used as context.

### Interactive UI
- Built with **Streamlit** for a seamless user experience.
- Real-time querying and context visualization.

## 🚀 Setup & Usage

### 1. Configuration
Create a `.env` file in this directory and add your Google API Key:
```env
GOOGLE_API_KEY=your_api_key_here
```

### 2. Ingestion
Initialize the vector database with the processed contract data:
```bash
python src/data_ingestion.py
```

### 3. Launch Application
Start the Streamlit server:
```bash
streamlit run src/app.py
```
