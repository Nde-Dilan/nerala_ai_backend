"""
PDF Processing Module

This module extracts text from PDFs, processes it into chunks, and generates 
embeddings stored in a vector database for semantic search and retrieval.
"""
import os
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import shutil
import hashlib

# Third-party imports
import tika
from tika import parser
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pdf_processing.log')
    ]
)
logger = logging.getLogger('pdf_processor')

# Configuration
CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": {
        "model_name": "sentence-transformers/all-MiniLM-l6-v2",
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": False}
    },
    "db_directory": "./vector_db",
    "allowed_extensions": [".pdf"],
    "max_file_size_mb": 26
}


class SecurityException(Exception):
    """Exception raised for security-related issues."""
    pass


class PDFProcessor:
    """Handles PDF text extraction, chunking, and embedding generation."""
    
    def __init__(self, config: Dict[str, Any] = CONFIG):
        """Initialize the PDF processor with configuration."""
        self.config = config
        self.text_splitter = CharacterTextSplitter(
            chunk_size=config["chunk_size"], 
            chunk_overlap=config["chunk_overlap"]
        )
        # Initialize Tika
        try:
            tika.initVM()
            logger.info("Tika initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tika: {e}")
            raise
            
        # Initialize embeddings model only when needed (lazy loading)
        self._embeddings_model = None
        
        # Ensure vector db directory exists
        os.makedirs(config["db_directory"], exist_ok=True)
        
    @property
    def embeddings_model(self):
        """Lazy-load the embeddings model."""
        if self._embeddings_model is None:
            try:
                logger.info("Initializing embedding model")
                self._embeddings_model = HuggingFaceEmbeddings(
                    model_name=self.config["embedding_model"]["model_name"],
                    model_kwargs=self.config["embedding_model"]["model_kwargs"],
                    encode_kwargs=self.config["embedding_model"]["encode_kwargs"]
                )
                logger.info("Embedding model initialized")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                raise
        return self._embeddings_model
            
    def validate_file(self, file_path: str) -> None:
        """
        Validate that the file is safe to process.
        
        Args:
            file_path: Path to the file to validate
            
        Raises:
            SecurityException: If the file fails security checks
        """
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Check file extension
        if path.suffix.lower() not in self.config["allowed_extensions"]:
            raise SecurityException(f"Unsupported file type: {path.suffix}")
            
        # Check file size
        max_size_bytes = self.config["max_file_size_mb"] * 1024 * 1024
        if path.stat().st_size > max_size_bytes:
            raise SecurityException(
                f"File exceeds maximum size of {self.config['max_file_size_mb']}MB"
            )
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            Various exceptions for file access or parsing issues
        """
        logger.info(f"Extracting text from {file_path}")
        self.validate_file(file_path)
        
        try:
            parsed_pdf = parser.from_file(file_path)
            
            if not parsed_pdf or "content" not in parsed_pdf or not parsed_pdf["content"]:
                logger.warning(f"No content extracted from {file_path}")
                return ""
                
            logger.info(f"Successfully extracted {len(parsed_pdf['content'])} characters")
            return parsed_pdf["content"]
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    def create_chunks(self, text: str) -> List[Document]:
        """
        Split text into manageable chunks.
        
        Args:
            text: The text to split
            
        Returns:
            List of document chunks
        """
        if not text:
            logger.warning("Empty text provided for chunking")
            return []
            
        logger.info(f"Splitting text into chunks (size={self.config['chunk_size']})")
        try:
            # Updated method takes texts as list and returns documents
            chunks = self.text_splitter.create_documents([text])
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            raise
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate a hash for the file to use as an identifier."""
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"Error generating file hash for {file_path}: {e}")
            raise
            
    def process_file(self, file_path: str) -> Optional[Chroma]:
        """
        Process a PDF file: extract text, create chunks, and generate embeddings.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Chroma vector store containing the embeddings
        """
        try:
            # Extract text
            text = self.extract_text(file_path)
            if not text:
                logger.warning(f"No text extracted from {file_path}")
                return None
                
            # Create chunks
            chunks = self.create_chunks(text)
            if not chunks:
                logger.warning(f"No chunks created from {file_path}")
                return None
                
            # Generate a unique persistent id based on file content
            file_hash = self.get_file_hash(file_path)
            persist_directory = os.path.join(self.config["db_directory"], file_hash)
            
            # Create vector store
            logger.info(f"Generating embeddings and storing in {persist_directory}")
            
            # Create directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)
            
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings_model,
                persist_directory=persist_directory
            )
            
            # Persist the vector store
            vector_store.persist()
            logger.info(f"Successfully processed {file_path}")
            
            return vector_store
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
            
    def query_document(self, file_path: str, query: str, k: int = 5) -> List[Document]:
        """
        Query a processed document with a natural language query.
        
        Args:
            file_path: Path to the PDF file that was processed
            query: The query string
            k: Number of results to return
            
        Returns:
            List of relevant document chunks
        """
        try:
            file_hash = self.get_file_hash(file_path)
            persist_directory = os.path.join(self.config["db_directory"], file_hash)
            
            if not os.path.exists(persist_directory):
                logger.warning(f"No vector store found for {file_path}")
                return []
                
            # Load the existing vector store
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings_model
            )
            
            # Query the vector store
            results = vector_store.similarity_search(query, k=k)
            logger.info(f"Query '{query}' returned {len(results)} results")
            
            return results
        except Exception as e:
            logger.error(f"Error querying document {file_path}: {e}")
            raise


def main():
    """Entry point for the PDF processing module."""
    try:
        # Example usage
        processor = PDFProcessor()
        file_path = "./Dictionnaire-Fulfulde-français-english-et-images.pdf"
        
        # Process the PDF
        vector_store = processor.process_file(file_path)
        
        if vector_store:
            logger.info("PDF processed successfully")
            
            # Example query (uncomment to use)
            # results = processor.query_document(file_path, "What is Fulfulde?")
            # for i, doc in enumerate(results):
            #     print(f"Result {i+1}:\n{doc.page_content}\n")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()
