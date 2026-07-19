import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from ingestion.embedding import EmbeddingManager
from ingestion.vectorstore import ChromaVectorStore
from retrieval.reranker import create_reranker
from retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """
    Orchestrates Hybrid Retrieval + Parent-Child Chunking + Cross-Encoder Reranking.

    Flow:
    1. Pull a broad pool with dense Chroma retrieval and sparse BM25.
    2. Expand matching child chunks into parent document contexts.
    3. Rerank the parent pool with a cross-encoder and return the final top k.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker=None,
        pool_size: int = 30,
        **kwargs: Any,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.pool_size = pool_size

    def retrieve(
        self,
        query: str,
        k: int = 3,
        use_reranker: bool = True,
        pool_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        if not query or not query.strip():
            return []

        retrieval_pool_size = pool_size or kwargs.pop("retrieval_pool_size", None)
        retrieval_pool_size = retrieval_pool_size or max(self.pool_size, k)

        retrieved = self.retriever.retrieve(query, k=retrieval_pool_size, **kwargs)
        if not retrieved:
            return []

        if use_reranker and self.reranker is not None:
            return self.reranker.rerank(query, retrieved, k=k)

        return retrieved[:k]

    def retrieve_with_metadata(
        self,
        query: str,
        k: int = 3,
        use_reranker: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        documents = self.retrieve(
            query, k=k, use_reranker=use_reranker, **kwargs
        )
        return {
            "query": query,
            "documents": documents,
            "document_count": len(documents),
            "retriever": "hybrid_bm25_dense_parent_child",
            "reranker": (
                getattr(self.reranker, "model_name", None)
                if use_reranker and self.reranker is not None
                else None
            ),
        }

    def retrieve_context(
        self,
        query: str,
        k: int = 3,
        use_reranker: bool = True,
        **kwargs: Any,
    ) -> str:
        documents = self.retrieve(
            query, k=k, use_reranker=use_reranker, **kwargs
        )
        return "\n\n---\n\n".join(
            f"[{index}] Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for index, doc in enumerate(documents, 1)
        )


def create_retrieval_pipeline(
    config,
    retriever_strategy: str = "hybrid",
    reranker_type: Optional[str] = None,
    k: int = 3,
    **kwargs: Any,
) -> RetrievalPipeline:
    if retriever_strategy.lower() != "hybrid":
        logger.info(
            "Retriever strategy '%s' requested; using hybrid BM25+dense retrieval.",
            retriever_strategy,
        )

    embedding_manager = EmbeddingManager(
        model_name=getattr(config, "embedding_model", "qwen3-embedding:0.6b"),
        base_url=getattr(config, "base_url", "http://localhost:11434"),
        dimensions=getattr(config, "embedding_dimension", 768),
    )
    vector_store = ChromaVectorStore(
        embedding_manager=embedding_manager,
        persist_directory=getattr(config, "chroma_persist_dir", "./chroma_db"),
        collection_name=getattr(config, "collection_name", "my_rag_collection"),
    )

    pool_size = kwargs.pop(
        "pool_size", getattr(config, "retrieval_pool_size", max(30, k * 10))
    )
    retriever = HybridRetriever(
        vector_store=vector_store,
        dense_weight=kwargs.pop("dense_weight", getattr(config, "dense_weight", 0.65)),
        sparse_weight=kwargs.pop(
            "sparse_weight", getattr(config, "sparse_weight", 0.35)
        ),
        pool_size=pool_size,
        parent_max_chars=kwargs.pop(
            "parent_max_chars", getattr(config, "parent_max_chars", 4000)
        ),
    )

    # The requested pipeline uses cross-encoder reranking by default.
    reranker_type = reranker_type or "cross_encoder"
    reranker = create_reranker(
        reranker_type,
        model_name=kwargs.pop(
            "reranker_model", getattr(config, "reranker_model", "qllama/bge-reranker-v2-m3")
        ),
    )

    return RetrievalPipeline(
        retriever=retriever,
        reranker=reranker,
        pool_size=pool_size,
    )
