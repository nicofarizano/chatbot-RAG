import logging
from pydantic import BaseModel
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain.docstore.document import Document
from langchain_core.runnables import Runnable
from config import get_settings

logger = logging.getLogger(__name__)

class Query(BaseModel):
    question: str
    openai_api_key: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str

class AIModel(Runnable):
    def __init__(self):
        self.settings = get_settings()
        self.embedding_model = GPT4AllEmbeddings()
        self.llm = OllamaLLM(base_url=self.settings.ollama_host, model=self.settings.model_name)
        self.vectorstore: FAISS | None = None

    def initialize_vectorstore(self, documents: List[Document]):
        """Initializes the vector store with the provided documents"""
        try:
            self.vectorstore = FAISS.from_documents(documents=documents, embedding=self.embedding_model)
            logger.info("✅ VectorStore initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing VectorStore: {e}")
            raise

    def generate_response(self, question: str, openai_api_key: Optional[str] = None) -> str:
        """Generates a response using OpenAI (if key is provided) or Ollama (default)."""
        if not self.vectorstore:
            return "⚠️  The system is not properly initialized."

        try:
            docs = self.vectorstore.similarity_search(question, k=5)
            if not docs:
                return "⚠️  No relevant documents were found for your question."

            return self._summarize_documents(docs, question, openai_api_key)
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return "⚠️  An error occurred while processing your question."

    def _summarize_documents(self, docs: List[Document], question: str, openai_api_key: Optional[str] = None) -> str:
        """Summarizes the relevant documents to answer the question"""
        combined_text = " ".join([doc.page_content for doc in docs])
        if len(combined_text) > self.settings.max_context_length:
            combined_text = combined_text[: self.settings.max_context_length]

        prompt = f"""
        You are a chatbot answering user questions based on the provided text. 
        Be concise. Prioritize relevant information and do not include content that is not related to the question.
        Question: {question}
        Relevant Information: {combined_text}
        Answer concisely in no more than {self.settings.max_words_response} words:
        """

        try:
            if openai_api_key and openai_api_key.strip():
                openai_llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4")
                logger.info("✅ Using OpenAI for response generation.")
                response = openai_llm.invoke(prompt)
                return str(response.content)

            logger.info("⚠️  No OpenAI key provided. Using Ollama instead.")
            return self.llm.invoke(prompt)
        
        except Exception as e:
            logger.error(f"❌ OpenAI Error: {e}")
            return "⚠️  Error: Could not connect to OpenAI. Please check your API Key or try again later."
        
    def invoke(self, input: dict) -> str:
        question = input.get("question", "")
        openai_api_key = input.get("openai_api_key", None)
        return self.generate_response(question, openai_api_key)