import logging
from pydantic import BaseModel
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_ollama import OllamaLLM
from langchain.docstore.document import Document
from config import get_settings

logger = logging.getLogger(__name__)

class Query(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

class AIModel:
    def __init__(self):
        self.settings = get_settings()
        self.embedding_model = GPT4AllEmbeddings()
        self.llm = OllamaLLM(model=self.settings.model_name)
        self.vectorstore: FAISS | None = None

    def initialize_vectorstore(self, documents: List[Document]):
        """Initializes the vector store with the provided documents"""
        try:
            self.vectorstore = FAISS.from_documents(
                documents=documents, embedding=self.embedding_model
            )
            logger.info("VectorStore initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing VectorStore: {e}")
            raise

    def generate_response(self, question: str) -> str:
        """Generates a response based on the question and relevant documents"""
        if not self.vectorstore:
            return "The system is not properly initialized."

        try:
            docs = self.vectorstore.similarity_search(question, k=5)
            if not docs:
                return "No relevant documents were found for your question."

            return self._summarize_documents(docs, question)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "An error occurred while processing your question."

    def _summarize_documents(self, docs: List[Document], question: str) -> str:
        """Summarizes the relevant documents to answer the question"""
        combined_text = " ".join([doc.page_content for doc in docs])
        if len(combined_text) > self.settings.max_context_length:
            combined_text = combined_text[: self.settings.max_context_length]

        prompt = f"""Below, you have a text. Answer the following question based on the text.
            Prioritize relevant information and avoid unrelated content.
            Respond only in English.
            Limit your response to no more than {self.settings.max_words_response} words.\n\n
            Question: {question}\n\n
            Text:\n{combined_text}
            """

        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Error in LLM: {e}")
            return "An error occurred while processing your question."
