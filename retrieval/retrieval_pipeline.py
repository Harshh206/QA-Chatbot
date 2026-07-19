from .retriever import AdvancedRetriever
from .reranker import get_reranker
from .query_processor import QueryProcessor
from ingestion.vectorstore import ChromaVectorStore
from ingestion.embedding import EmbeddingManager
from config import config
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Complete retrieval pipeline with processing, retrieval, and reranking"""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_manager: EmbeddingManager,
        retriever_strategy: str = "simple",
        reranker_type: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize retrieval pipeline

        Args:
            vector_store: ChromaVectorStore instance
            embedding_manager: EmbeddingManager instance
            retriever_strategy: Retrieval strategy name
            reranker_type: Reranker type ('cross_encoder', 'llm', 'rrf')
            **kwargs: Additional parameters
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.query_processor = QueryProcessor(embedding_manager)

        # Extract k values from kwargs
        self.k = kwargs.get("k", 10)
        self.retrieve_k = kwargs.get("retrieve_k", self.k * 2)

        # Remove k from kwargs before passing to retriever
        retriever_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["k", "retrieve_k", "reranker_type"]
        }

        # Initialize retriever
        self.retriever = AdvancedRetriever(
            vector_store,
            embedding_manager,
            strategy=retriever_strategy,
            **retriever_kwargs,
        )

        # Initialize reranker if specified
        self.reranker = None
        if reranker_type:
            # Filter kwargs for reranker (remove pipeline-specific params)
            reranker_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "k",
                    "retrieve_k",
                    "strategy",
                    "retriever_strategy",
                    "reranker_type",
                ]
            }
            self.reranker = get_reranker(reranker_type, **reranker_kwargs)

    def retrieve(
        self, query: str, k: Optional[int] = None, use_reranker: bool = True, **kwargs
    ) -> List[Document]:
        """
        Retrieve relevant documents

        Args:
            query: Search query
            k: Number of documents to return
            use_reranker: Whether to use reranker
            **kwargs: Additional parameters

        Returns:
            List of relevant documents
        """
        # Process query
        processed = self.query_processor.preprocess_query(query)
        logger.info(f"Retrieving for query: {processed}")

        # Determine k values
        final_k = k or self.k
        retrieve_k = kwargs.get("retrieve_k", self.retrieve_k)

        # Retrieve more documents than needed for reranking
        if use_reranker and self.reranker:
            retrieve_k = max(retrieve_k, final_k * 2)

        # Remove conflicting kwargs
        search_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["retrieve_k", "per_query_k"]
        }

        # Retrieve documents
        documents = self.retriever.retrieve(processed, k=retrieve_k, **search_kwargs)

        # Apply reranking
        if use_reranker and self.reranker and documents:
            try:
                documents = self.reranker.rerank(processed, documents)
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
            # Limit to final k
            documents = documents[:final_k]
        else:
            # Limit to k
            documents = documents[:final_k]

        logger.info(f"Retrieved {len(documents)} documents")
        return documents

    def retrieve_with_metadata(
        self, query: str, k: Optional[int] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with additional metadata

        Returns:
            List of dictionaries with document and metadata
        """
        documents = self.retrieve(query, k=k, **kwargs)

        results = []
        for i, doc in enumerate(documents):
            results.append(
                {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("score", None),
                }
            )

        return results

    def retrieve_context(
        self,
        query: str,
        k: Optional[int] = None,
        include_sources: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Retrieve context formatted for RAG

        Returns:
            Dictionary with context and sources
        """
        documents = self.retrieve(query, k=k, **kwargs)

        context_parts = []
        sources = []

        for i, doc in enumerate(documents):
            context_parts.append(doc.page_content)
            if include_sources:
                sources.append(
                    {
                        "content": doc.page_content[:200] + "...",
                        "source": doc.metadata.get("source", "Unknown"),
                        "rank": i + 1,
                    }
                )

        context = "\n\n".join(context_parts)

        return {
            "query": query,
            "context": context,
            "sources": sources,
            "document_count": len(documents),
        }

    async def aretrieve(
        self, query: str, k: Optional[int] = None, **kwargs
    ) -> List[Document]:
        """Async retrieval"""
        processed = self.query_processor.preprocess_query(query)
        logger.info(f"Async retrieving for query: {processed}")

        final_k = k or self.k
        documents = await self.retriever.aretrieve(processed, k=final_k, **kwargs)

        if self.reranker and documents:
            documents = self.reranker.rerank(processed, documents)
            documents = documents[:final_k]

        return documents

    def change_strategy(self, strategy: str):
        """Change retrieval strategy"""
        self.retriever.change_strategy(strategy)
        logger.info(f"Changed retrieval strategy to: {strategy}")


def create_retrieval_pipeline(
    config,
    retriever_strategy: str = "simple",
    reranker_type: Optional[str] = None,
    **kwargs,
) -> RetrievalPipeline:
    """Create and configure a retrieval pipeline"""
    # Initialize components
    embedding_manager = EmbeddingManager(
        model_name=config.embedding_model, base_url=config.base_url
    )

    vector_store = ChromaVectorStore(
        embedding_manager=embedding_manager,
        persist_directory=config.chroma_persist_dir,
        collection_name=config.collection_name,
    )

    # Create pipeline
    pipeline = RetrievalPipeline(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        retriever_strategy=retriever_strategy,
        reranker_type=reranker_type,
        **kwargs,
    )

    return pipeline
