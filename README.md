# NLP Legal Contract Analysis Pipeline

An end-to-end pipeline for analyzing English legal contracts, featuring syntax parsing, information extraction, and a Retrieval-Augmented Generation (RAG) Question Answering system.

## 📁 Project Structure

- **[BTL1](./BTL1)**: **Syntax Analysis & Preprocessing**. Handles clause splitting, noun phrase chunking, and dependency parsing using spaCy.
- **[BTL2](./BTL2)**: **Information Extraction & Semantics**. Features custom NER (Named Entity Recognition), SRL (Semantic Role Labeling), and Intent Classification using DistilBERT and spaCy.
- **[BTL3](./BTL3)**: **Contract QA Application**. A RAG-based chatbot interface built with LangChain, ChromaDB, and Google Gemini API.
- **`main.ipynb`**: Integrated notebook for experimental runs and visualization.
- **`requirements.txt`**: Global dependencies for the project.

## 📊 Data Management

The project follows a structured data flow where outputs from earlier assignments serve as inputs for subsequent ones.

### BTL1: Preprocessing
- **Input**: `BTL1/input/raw_contracts.txt` (Unstructured legal text)
- **Outputs**:
  - `BTL1/output/clauses.txt`: Cleaned independent clauses.
  - `BTL1/output/chunks.txt`: Noun phrase chunks (IOB format).
  - `BTL1/output/dependency.json`: Syntactic dependency trees.

### BTL2: Extraction & Semantics
- **Training Data**: `BTL2/data/` (Contains annotated JSON files for NER and Intent training).
- **Inference Input**: Uses `BTL1/output/clauses.txt`.
- **Outputs**:
  - `BTL2/output/ner_results.json`: Extracted legal entities.
  - `BTL2/output/srl_results.json`: Semantic roles mapped to clauses.
  - `BTL2/output/intent_classification.txt`: Classified legal intents.

### BTL3: Question Answering (RAG)
- **Source Data**: Consumes data from `BTL1/output/` and `BTL2/output/` to build the knowledge base.
- **Vector Database**: `BTL3/chroma_db/` (Generated after running `data_ingestion.py`).


## ⚙️ Installation

To set up the core environments for BTL1 and BTL2:

```bash
# Install core dependencies
pip install spacy scikit-learn pandas matplotlib seaborn
# Download spaCy model
python -m spacy download en_core_web_sm
```

*Note: BTL3 requires additional dependencies. See the [BTL3 README](./BTL3/README.md) for details.*

## 🚀 Usage Guide

### Option 1: Quick Start with Jupyter Notebook (Recommended)
For an integrated experience with visualizations and step-by-step execution for **BTL1** and **BTL2**, use the provided notebook at the root:
- **`main.ipynb`**: Contains the full pipeline from preprocessing to semantic analysis.

### Option 2: Command Line Interface (CLI)

#### 1. Preprocessing (BTL1)
Process raw contract text (`BTL1/input/raw_contracts.txt`) into structured clauses:
```bash
cd BTL1
python src/main.py
```

#### 2. Extraction & Training (BTL2)
Perform entity extraction and intent classification on processed clauses:
```bash
cd BTL2
# Full Run (Train + Inference)
python src/main.py 

# Quick Inference Only (Skip DistilBERT training)
python src/main.py --no-transformer
```

### 3. Interactive Chatbot (BTL3)
The RAG-powered chatbot requires a separate setup and execution flow.

**Step A: Configure API Key**
Update `.env` in the `BTL3` directory with your Google Gemini API Key:
```env
GOOGLE_API_KEY=your_api_key_here
```

**Step B: Ingest Data (First run only)**
```bash
cd BTL3
python src/data_ingestion.py
```

**Step C: Start UI**
```bash
streamlit run src/app.py
```


## 📊 Results & Visualization
Detailed performance metrics and visualizations for NER, SRL, and Intent Classification are available in the respective assignment folders and integrated into the final report.