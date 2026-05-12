import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class RAGEngine:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        persist_directory = os.path.join(base_dir, "chroma_db")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load Vector Store
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        # Setup Retriever with MMR Strategy
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 3, 'fetch_k': 10}
        )
        
        # Setup LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0,
            max_retries=2
        )
        
        # Setup Strict Prompt
        system_prompt = (
            "You are a professional legal assistant. Use the FOLLOWING CONTEXT to answer the user's question.\n"
            "If the context does not contain the information to answer, CLEARLY STATE THAT YOU DO NOT KNOW "
            "or that the information is not in the contract. ABSOLUTELY DO NOT invent an answer.\n"
            "At the end of your answer, YOU MUST cite the original text used as the source in the format: '(Source: [Original contract clause])'.\n\n"
            "Context: {context}\n"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        
        # Create LCEL Chain
        self.rag_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        return "\n\n".join(f"[{doc.metadata.get('source', 'Unknown')}]: {doc.page_content}" for doc in docs)
        
    def ask(self, query: str) -> str:
        try:
            return self.rag_chain.invoke(query)
        except Exception as e:
            return f"An error occurred while generating the answer: {str(e)}"
