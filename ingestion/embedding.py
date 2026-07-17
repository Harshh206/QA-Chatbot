from langchain_ollama import OllamaEmbeddings
from typing import List, Optional
from config import config
import logging

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages Ollama embedding models and operations"""

    def __init__(
        self,
        model_name: str = config.embedding_model,
        base_url: str = config.ollama_base_url,
        dimensions: int = config.embedding_dimension,
    ):
      
        self.model_name = model_name or config.embedding_model
        self.base_url = base_url
        self.dimensions = dimensions
        self.embeddings = self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize Ollama embeddings"""
        try:
            return OllamaEmbeddings(
                model=self.model_name,
                base_url=self.base_url,
                dimensions=self.dimensions,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embeddings: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.warning("No texts provided for embedding")
            return []

        try:
            logger.info(
                f"Embedding {len(texts)} documents using Ollama model: {self.model_name}"
            )
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query

        Args:
            query: Query string to embed

        Returns:
            Embedding vector
        """
        if not query:
            logger.warning("Empty query provided for embedding")
            return []

        try:
            return self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

    def embed_chunks(self, chunks: List) -> List:
        """Embed document chunks and return them with embeddings

        Args:
            chunks: List of Document chunks

        Returns:
            List of Document chunks with embeddings added to metadata
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return []

        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embed_documents(texts)

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
           #
            chunk.metadata["embedding_model"] = self.model_name

        logger.info(f"Added embeddings to {len(chunks)} chunks")
        return chunks

    def get_embedding_dimension(self) -> Optional[int]:
        """Get the dimension of embeddings from the current model

        Returns:
            Embedding dimension or None if unavailable
        """
        try:
            # Test with a sample text to get dimension
            test_embedding = self.embed_query("test")
            return len(test_embedding) if test_embedding else None
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
            return None

    def validate_connection(self) -> bool:
        """Validate connection to Ollama server

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try embedding a simple text to check connection
            self.embed_query("connection test")
            logger.info(f"Successfully connected to Ollama at {self.base_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
