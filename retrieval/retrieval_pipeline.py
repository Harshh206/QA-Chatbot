from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
import logging
import re

from .retriever import HybridRetriever
from .reranker import OllamaCrossEncoderReranker
from ingestion.vectorstore import ChromaVectorStore
from ingestion.embedding import EmbeddingManager

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """
    Full retrieval pipeline:
        User Query
            ├── Dense Vector Search (qwen3-embedding:0.6b)
            └── Sparse BM25 Search
                    │
                    ▼
            Combined Pool (RRF Fusion)
                    │
                    ▼
            Cross-Encoder Reranker (bge-reranker-v2-m3)
                    │
                    ▼
            Dynamic Score Filter (> threshold)
                    │
                    ▼
            Top N Hyper-Exact Chunks → LLM
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_manager: EmbeddingManager,
        top_k: int = 5,
        score_threshold: float = 0.70,
        reranker_model: str = "qllama/bge-reranker-v2-m3:latest",
        base_url: str = "http://localhost:11434",
        rrf_k: int = 60,
    ):
        self.top_k = top_k
        self.score_threshold = score_threshold

        self.retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_manager=embedding_manager,
            rrf_k=rrf_k,
        )

        self.reranker = OllamaCrossEncoderReranker(
            model_name=reranker_model,
            score_threshold=score_threshold,
        )

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        final_k = k or self.top_k

        logger.info(f"Retrieving for query: {query}")

        candidates = self.retriever.retrieve(query, k=final_k * 3)
        logger.info(f"Hybrid retrieval returned {len(candidates)} candidates")

        if not candidates:
            return []

        reranked = self.reranker.rerank(query, candidates)
        logger.info(f"Reranker returned {len(reranked)} documents")

        return reranked[:final_k]

    def retrieve_with_scores(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        final_k = k or self.top_k
        documents = self.retrieve(query, k=final_k)

        results = []
        for i, doc in enumerate(documents):
            results.append({
                "rank": i + 1,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "Unknown"),
            })
        return results

    def retrieve_context(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        documents = self.retrieve(query, k=k)

        context_parts = []
        sources = []
        for i, doc in enumerate(documents):
            context_parts.append(doc.page_content)
            sources.append({
                "rank": i + 1,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
            })

        return {
            "query": query,
            "context": "\n\n".join(context_parts),
            "sources": sources,
            "document_count": len(documents),
        }

    def change_threshold(self, threshold: float):
        self.score_threshold = threshold
        self.reranker.score_threshold = threshold
        logger.info(f"Score threshold changed to {threshold}")


def create_retrieval_pipeline(
    config,
    top_k: int = 5,
    score_threshold: float = 0.70,
    reranker_model: str = "qllama/bge-reranker-v2-m3:latest",
    **kwargs,
) -> RetrievalPipeline:
    embedding_manager = EmbeddingManager(
        model_name=config.embedding_model, base_url=config.base_url
    )

    vector_store = ChromaVectorStore(
        embedding_manager=embedding_manager,
        persist_directory=config.chroma_persist_dir,
        collection_name=config.collection_name,
    )

    return RetrievalPipeline(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        top_k=top_k,
        score_threshold=score_threshold,
        reranker_model=reranker_model,
        base_url=config.base_url,
        **kwargs,
    )
