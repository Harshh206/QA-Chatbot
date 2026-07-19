from langchain_core.documents import Document
from typing import List, Dict, Tuple, Optional
from math import log
import logging

from ingestion.vectorstore import ChromaVectorStore
from ingestion.embedding import EmbeddingManager

logger = logging.getLogger(__name__)


class BM25SparseRetriever:
    """BM25 sparse retriever operating on the vector store collection"""

    def __init__(self, vector_store: ChromaVectorStore, k1: float = 1.5, b: float = 0.75):
        self.vector_store = vector_store
        self.k1 = k1
        self.b = b

    def retrieve(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        all_docs = self.vector_store.vector_store.get(include=["documents", "metadatas"])
        documents_text = all_docs.get("documents", [])
        metadatas = all_docs.get("metadatas", [])

        if not documents_text:
            return []

        tokenized_docs = [doc.lower().split() for doc in documents_text]
        doc_count = len(tokenized_docs)
        avg_dl = sum(len(d) for d in tokenized_docs) / doc_count if doc_count else 1

        tokenized_query = query.lower().split()

        doc_freqs = {}
        for doc in tokenized_docs:
            seen = set()
            for term in doc:
                if term not in seen:
                    doc_freqs[term] = doc_freqs.get(term, 0) + 1
                    seen.add(term)

        scores = []
        for i, doc_tokens in enumerate(tokenized_docs):
            score = 0.0
            dl = len(doc_tokens)
            tf_map = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for term in tokenized_query:
                if term not in tf_map:
                    continue
                tf = tf_map[term]
                df = doc_freqs.get(term, 0)
                if df == 0:
                    continue
                idf = log((doc_count - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / avg_dl))
                score += idf * tf_norm

            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:k]:
            doc = Document(
                page_content=documents_text[idx],
                metadata=metadatas[idx] if idx < len(metadatas) else {},
            )
            results.append((doc, score))

        return results


class DenseRetriever:
    """Dense vector retriever using qwen3-embedding:0.6b"""

    def __init__(self, vector_store: ChromaVectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        results = self.vector_store.search_with_score(query, k=k)
        return [(doc, float(score)) for doc, score in results]


class HybridRetriever:
    """
    Hybrid retriever combining BM25 sparse + dense vector search.
    Uses Reciprocal Rank Fusion to merge ranked lists.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_manager: EmbeddingManager,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.rrf_k = rrf_k
        self.bm25 = BM25SparseRetriever(vector_store)
        self.dense = DenseRetriever(vector_store, embedding_manager)

    def retrieve(self, query: str, k: int = 20) -> List[Document]:
        retrieve_k = max(k * 3, 30)

        sparse_results = self.bm25.retrieve(query, k=retrieve_k)
        dense_results = self.dense.retrieve(query, k=retrieve_k)

        merged = self._rrf_fusion(sparse_results, dense_results)
        return merged[:k]

    def _rrf_fusion(
        self,
        sparse_results: List[Tuple[Document, float]],
        dense_results: List[Tuple[Document, float]],
    ) -> List[Document]:
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, (doc, _) in enumerate(sparse_results):
            key = doc.page_content[:200]
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0.0) + rrf_score
            if key not in doc_map:
                doc_map[key] = doc

        for rank, (doc, _) in enumerate(dense_results):
            key = doc.page_content[:200]
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0.0) + rrf_score
            if key not in doc_map:
                doc_map[key] = doc

        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        return [doc_map[key] for key in sorted_keys]
