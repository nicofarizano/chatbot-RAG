import logging
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
from config import get_settings

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self):
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    def load_web_documents(self) -> List[Document]:
        """Loads documents from URLs"""
        try:
            logger.info("Loading web documents...")
            web_loader = WebBaseLoader(self.settings.urls)
            web_data = web_loader.load()
            return self.text_splitter.split_documents(web_data)
        except Exception as e:
            logger.error(f"Error loading web documents: {e}")
            return []

    def load_pdf_documents(self) -> List[Document]:
        """Loads documents from PDF file"""
        try:
            logger.info("Loading PDF documents...")
            pdf_loader = PyPDFLoader(self.settings.pdf_path)
            pdf_data = pdf_loader.load()
            return self.text_splitter.split_documents(pdf_data)
        except Exception as e:
            logger.error(f"Error loading PDF documents: {e}")
            return []

    def load_all_documents(self) -> List[Document]:
        """Loads and combines all documents"""
        web_docs = self.load_web_documents()
        pdf_docs = self.load_pdf_documents()
        all_docs = web_docs + pdf_docs
        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs