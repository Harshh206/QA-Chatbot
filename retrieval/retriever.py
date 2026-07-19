import logging
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Small dependency-free tokenizer for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


def _document_key(document: Document) -> str:
    metadata = document.metadata or {}
    return str(
        metadata.get("id")
        or metadata.get("chunk_id")
        or metadata.get("content_hash")
        or f"{metadata.get('source', '')}:{metadata.get('page', '')}:{hash(document.page_content)}"
    )


def _parent_key(document: Document) -> str:
    metadata = document.metadata or {}
    if metadata.get("parent_id"):
        return str(metadata["parent_id"])
    if metadata.get("doc_id"):
        return str(metadata["doc_id"])
    if metadata.get("source") is not None and metadata.get("page") is not None:
        return f"{metadata['source']}::page:{metadata['page']}"
    if metadata.get("source"):
        return str(metadata["source"])
    return _document_key(document)


def _min_max(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if math.isclose(low, high):
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


class BM25Retriever:
    """Simple sparse BM25 retriever over the documents stored in Chroma."""

    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [_tokenize(doc.page_content) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.doc_freqs = self._build_doc_freqs(self.doc_tokens)
        self.doc_count = len(documents)

    @staticmethod
    def _build_doc_freqs(doc_tokens: Iterable[List[str]]) -> Counter:
        doc_freqs = Counter()
        for tokens in doc_tokens:
            doc_freqs.update(set(tokens))
        return doc_freqs

    def _idf(self, term: str) -> float:
        doc_freq = self.doc_freqs.get(term, 0)
        return math.log(1 + (self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

    def score(self, query: str) -> Dict[str, float]:
        query_terms = _tokenize(query)
        if not query_terms or not self.documents:
            return {}

        scores: Dict[str, float] = {}
        for index, tokens in enumerate(self.doc_tokens):
            if not tokens:
                continue

            term_counts = Counter(tokens)
            doc_length = self.doc_lengths[index]
            score = 0.0

            for term in query_terms:
                term_frequency = term_counts.get(term, 0)
                if term_frequency == 0:
                    continue

                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / (self.avg_doc_length or 1)
                )
                score += self._idf(term) * (
                    term_frequency * (self.k1 + 1) / denominator
                )

            if score > 0:
                scores[_document_key(self.documents[index])] = score

        return scores

    def retrieve(self, query: str, k: int) -> List[Tuple[Document, float]]:
        scores = self.score(query)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        document_by_key = {_document_key(doc): doc for doc in self.documents}
        return [(document_by_key[key], score) for key, score in ranked]


class HybridRetriever:
    """
    Combines dense vector retrieval with sparse BM25, then expands matching child
    chunks into parent document contexts.
    """

    def __init__(
        self,
        vector_store,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
        pool_size: int = 30,
        parent_max_chars: int = 4000,
    ):
        self.vector_store = vector_store
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.pool_size = pool_size
        self.parent_max_chars = parent_max_chars
        self._documents: Optional[List[Document]] = None
        self._bm25: Optional[BM25Retriever] = None

    @property
    def documents(self) -> List[Document]:
        if self._documents is None:
            self._documents = self._load_documents()
        return self._documents

    @property
    def bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            self._bm25 = BM25Retriever(self.documents)
        return self._bm25

    def _load_documents(self) -> List[Document]:
        try:
            raw = self.vector_store.vector_store._collection.get(
                include=["documents", "metadatas"]
            )
        except Exception as exc:
            logger.error(f"Could not load documents from Chroma for BM25: {exc}")
            return []

        ids = raw.get("ids", []) or []
        texts = raw.get("documents", []) or []
        metadatas = raw.get("metadatas", []) or []
        documents = []

        for index, text in enumerate(texts):
            metadata = dict(metadatas[index] or {})
            metadata.setdefault("id", ids[index] if index < len(ids) else str(index))
            metadata.setdefault("child_index", index)
            documents.append(Document(page_content=text or "", metadata=metadata))

        logger.info(f"Loaded {len(documents)} chunks for sparse retrieval")
        return documents

    def _dense_scores(self, query: str, k: int) -> Dict[str, Tuple[Document, float]]:
        dense_results = self.vector_store.search_with_score(query, k=k)
        scores: Dict[str, Tuple[Document, float]] = {}

        for document, distance in dense_results:
            key = _document_key(document)
            relevance = 1 / (1 + max(float(distance), 0.0))
            scores[key] = (document, relevance)

        return scores

    def _sparse_scores(self, query: str, k: int) -> Dict[str, Tuple[Document, float]]:
        return {
            _document_key(document): (document, score)
            for document, score in self.bm25.retrieve(query, k=k)
        }

    def _combine_scores(
        self,
        dense_scores: Dict[str, Tuple[Document, float]],
        sparse_scores: Dict[str, Tuple[Document, float]],
    ) -> List[Document]:
        dense_norm = _min_max(
            {key: score for key, (_, score) in dense_scores.items()}
        )
        sparse_norm = _min_max(
            {key: score for key, (_, score) in sparse_scores.items()}
        )
        all_keys = set(dense_scores) | set(sparse_scores)
        ranked_documents = []

        for key in all_keys:
            document = dense_scores.get(key, sparse_scores.get(key))[0]
            score = (
                self.dense_weight * dense_norm.get(key, 0.0)
                + self.sparse_weight * sparse_norm.get(key, 0.0)
            )
            metadata = dict(document.metadata or {})
            metadata["dense_score"] = dense_norm.get(key, 0.0)
            metadata["bm25_score"] = sparse_norm.get(key, 0.0)
            metadata["hybrid_score"] = score
            metadata["retrieval_stage"] = "hybrid_child"
            ranked_documents.append(
                Document(page_content=document.page_content, metadata=metadata)
            )

        ranked_documents.sort(
            key=lambda document: document.metadata.get("hybrid_score", 0.0),
            reverse=True,
        )
        return ranked_documents

    def _expand_to_parent_documents(self, child_documents: List[Document]) -> List[Document]:
        corpus_by_parent: Dict[str, List[Document]] = defaultdict(list)
        for document in self.documents:
            corpus_by_parent[_parent_key(document)].append(document)

        parent_results: Dict[str, Document] = {}
        for rank, child in enumerate(child_documents, 1):
            key = _parent_key(child)
            if key in parent_results:
                continue

            siblings = sorted(
                corpus_by_parent.get(key, [child]),
                key=lambda doc: doc.metadata.get("child_index", 0),
            )
            parent_text_parts = []
            total_chars = 0
            for sibling in siblings:
                text = sibling.page_content.strip()
                if not text:
                    continue
                if total_chars + len(text) > self.parent_max_chars:
                    remaining = self.parent_max_chars - total_chars
                    if remaining > 200:
                        parent_text_parts.append(text[:remaining])
                    break
                parent_text_parts.append(text)
                total_chars += len(text)

            metadata = dict(child.metadata or {})
            metadata["parent_id"] = key
            metadata["child_rank"] = rank
            metadata["retrieval_stage"] = "parent_expanded"
            metadata["child_chunks_in_parent"] = len(siblings)
            parent_results[key] = Document(
                page_content="\n\n".join(parent_text_parts) or child.page_content,
                metadata=metadata,
            )

        return list(parent_results.values())

    def retrieve(self, query: str, k: Optional[int] = None, **kwargs: Any) -> List[Document]:
        pool_size = k or self.pool_size
        dense_scores = self._dense_scores(query, pool_size)
        sparse_scores = self._sparse_scores(query, pool_size)

        if not dense_scores and not sparse_scores:
            logger.warning("Hybrid retrieval found no documents")
            return []

        child_pool = self._combine_scores(dense_scores, sparse_scores)
        return self._expand_to_parent_documents(child_pool)[:pool_size]


AdvancedRetriever = HybridRetriever
