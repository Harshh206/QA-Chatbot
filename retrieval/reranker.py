from langchain_core.documents import Document
from typing import List, Optional
import logging
import re
from config import config

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-Encoder Reranker using sentence-transformers.
    Uses BAAI/bge-reranker-v2-m3 from HuggingFace (local).
    """

    def __init__(
        self,
        model_name: str = config.reranker_model,
        score_threshold: float = config.threshold,
        max_docs: int = 20,
        **kwargs,
    ):
        self.model_name = model_name
        self.score_threshold = score_threshold
        self.max_docs = max_docs
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded cross-encoder: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Using fallback scoring.")
            self.model = None
        except Exception as e:
            logger.warning(f"Could not load cross-encoder: {e}. Using fallback.")
            self.model = None

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        docs_to_rerank = documents[:self.max_docs]
        remaining = documents[self.max_docs:]

        if self.model is None:
            return self._fallback_rerank(query, docs_to_rerank) + remaining

        try:
            pairs = [[query, doc.page_content[:512]] for doc in docs_to_rerank]
            scores = self.model.predict(pairs)

            scored_docs = list(zip(docs_to_rerank, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            results = [doc for doc, score in scored_docs if float(score) >= self.score_threshold]

            if not results and scored_docs:
                results = [scored_docs[0][0]]

            logger.info(
                f"Reranked {len(docs_to_rerank)} docs, kept {len(results)} "
                f"(threshold={self.score_threshold})"
            )

            return results + remaining

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return self._fallback_rerank(query, docs_to_rerank) + remaining

    def _fallback_rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Keyword overlap fallback when model unavailable"""
        query_words = set(query.lower().split())
        scored = []
        for doc in documents:
            content_words = set(doc.page_content.lower().split())
            if not query_words:
                scored.append((doc, 0.0))
            else:
                score = len(query_words & content_words) / len(query_words)
                scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored]
