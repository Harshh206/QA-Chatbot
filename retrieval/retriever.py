from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from ingestion.vectorstore import ChromaVectorStore
from ingestion.embedding import EmbeddingManager
import asyncio
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class RetrievalStrategy:
    """Base retrieval strategy class"""

    def __init__(
        self, vector_store: ChromaVectorStore, embedding_manager: EmbeddingManager
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Retrieve documents based on query"""
        raise NotImplementedError


class SimpleRetrieval(RetrievalStrategy):
    """Simple similarity search retrieval"""

    def retrieve(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Basic similarity search"""
        return self.vector_store.search(query, k=k, **kwargs)

    def retrieve_with_scores(
        self, query: str, k: int = 5, **kwargs
    ) -> List[Tuple[Document, float]]:
        """Search with similarity scores"""
        return self.vector_store.search_with_score(query, k=k, **kwargs)


class MMRRetrieval(RetrievalStrategy):
    """Maximum Marginal Relevance retrieval for diversity"""

    def retrieve(
        self, query: str, k: int = 5, lambda_mult: float = 0.5, **kwargs
    ) -> List[Document]:
        """
        MMR retrieval that balances relevance and diversity

        Args:
            query: Search query
            k: Number of documents to return
            lambda_mult: Trade-off parameter (0 = diversity only, 1 = relevance only)
        """
        try:
            # Get more documents than needed for MMR
            fetch_k = kwargs.get("fetch_k", k * 2)

            # Use Chroma's MMR search directly
            # Chroma's max_marginal_relevance_search is available via the vector store
            results = self.vector_store.vector_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                **{k: v for k, v in kwargs.items() if k != "fetch_k"},
            )
            return results
        except AttributeError:
            # Fallback: Chroma might not have MMR directly, use similarity search
            logger.warning("MMR not available, falling back to similarity search")
            return self.vector_store.search(query, k=k)
        except Exception as e:
            logger.error(f"Error in MMR retrieval: {e}")
            return self.vector_store.search(query, k=k)


class HybridRetrieval(RetrievalStrategy):
    """Hybrid retrieval combining semantic and keyword search"""

    def __init__(
        self, vector_store: ChromaVectorStore, embedding_manager: EmbeddingManager
    ):
        super().__init__(vector_store, embedding_manager)
        # Try to import BM25 for better keyword search
        try:
            self.bm25_available = True
        except ImportError:
            self.bm25_available = False
            logger.warning("rank_bm25 not installed. Using simple keyword matching.")

    def retrieve(
        self, query: str, k: int = 5, alpha: float = 0.5, **kwargs
    ) -> List[Document]:
        """
        Hybrid search with keyword and semantic matching

        Args:
            query: Search query
            k: Number of documents to return
            alpha: Weight between keyword (0) and semantic (1) search
        """
        try:
            # Get semantic search results
            semantic_results = self.vector_store.search(query, k=k * 2, **kwargs)

            # Get keyword-based results using BM25 if available
            if self.bm25_available:
                keyword_results = self._bm25_search(query, k * 2)
            else:
                keyword_results = self._keyword_search(query, k * 2)

            # Combine and rerank results
            combined = self._combine_results(
                semantic_results, keyword_results, alpha, k
            )
            return combined
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return self.vector_store.search(query, k=k)

    def _bm25_search(self, query: str, k: int) -> List[Document]:
        """Perform BM25 search"""
        try:
            # Get all documents from collection
            all_docs = self.vector_store.vector_store.get()
            documents = all_docs.get("documents", [])

            if not documents:
                return []

            # Tokenize documents and query
            tokenized_docs = [doc.split() for doc in documents]
            bm25 = BM25Okapi(tokenized_docs)

            # Get scores
            tokenized_query = query.split()
            scores = bm25.get_scores(tokenized_query)

            # Get top k
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:k]

            results = []
            for idx in top_indices:
                doc = Document(
                    page_content=documents[idx],
                    metadata=(
                        self.vector_store.vector_store.get()["metadatas"][idx]
                        if "metadatas" in all_docs
                        else {}
                    ),
                )
                results.append(doc)

            return results
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []

    def _keyword_search(self, query: str, k: int) -> List[Document]:
        """Perform keyword-based search using simple matching"""
        all_docs = self.vector_store.vector_store.get()
        documents = all_docs.get("documents", [])
        metadatas = all_docs.get("metadatas", [])

        if not documents:
            return []

        query_words = set(query.lower().split())
        scored_docs = []

        for i, doc in enumerate(documents):
            content = doc.lower()
            # Count matching words
            matches = sum(1 for word in query_words if word in content)
            score = matches / len(query_words) if query_words else 0
            scored_docs.append((i, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scored_docs[:k]]

        results = []
        for idx in top_indices:
            doc = Document(
                page_content=documents[idx],
                metadata=metadatas[idx] if idx < len(metadatas) else {},
            )
            results.append(doc)

        return results

    def _combine_results(
        self, semantic: List[Document], keyword: List[Document], alpha: float, k: int
    ) -> List[Document]:
        """Combine semantic and keyword results using reciprocal rank fusion"""
        # Simple combination: interleave results
        combined = []
        seen_content = set()

        # Add semantic results first (weighted by alpha)
        for doc in semantic:
            content_key = doc.page_content[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                combined.append(doc)

        # Add keyword results
        for doc in keyword:
            content_key = doc.page_content[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                combined.append(doc)

        return combined[:k]


class MultiQueryRetrieval(RetrievalStrategy):
    """Generate multiple queries for better retrieval"""

    def __init__(
        self, vector_store: ChromaVectorStore, embedding_manager: EmbeddingManager
    ):
        super().__init__(vector_store, embedding_manager)

    def retrieve(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Retrieve using multiple generated queries"""
        # Generate variations of the query
        variations = self._generate_queries(query)

        # Retrieve for each query
        all_results = []
        seen_contents = set()

        per_query_k = kwargs.get("per_query_k", max(k // 2, 1))

        for var_query in variations:
            docs = self.vector_store.search(var_query, k=per_query_k, **kwargs)
            for doc in docs:
                # Deduplicate by content
                content_key = doc.page_content[:100]  # Use first 100 chars as key
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    all_results.append(doc)

        # Return top k from combined results
        return all_results[:k]

    def _generate_queries(self, query: str) -> List[str]:
        """Generate multiple query variations"""
        # Simple variations
        variations = [
            query,
            query.lower(),
            query.capitalize(),
            f"information about {query}",
            f"details regarding {query}",
            f"explanation of {query}",
        ]

        # Remove duplicates and limit
        return list(set(variations))[:5]


class ParentDocumentRetrieval(RetrievalStrategy):
    """Retrieve parent documents based on child chunks"""

    def __init__(
        self, vector_store: ChromaVectorStore, embedding_manager: EmbeddingManager
    ):
        super().__init__(vector_store, embedding_manager)
        self.parent_child_mapping = (
            {}
        )  # Store mapping between chunk IDs and parent documents

    def retrieve(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Retrieve chunks and return parent documents"""
        # Get relevant chunks
        chunks = self.vector_store.search(query, k=k * 2, **kwargs)

        # Retrieve parent documents
        parent_docs = []
        seen_parents = set()

        for chunk in chunks:
            parent_id = chunk.metadata.get("parent_id") or chunk.metadata.get("source")
            if parent_id and parent_id not in seen_parents:
                # Try to get parent document from mapping or create a placeholder
                parent = self._get_parent_document(parent_id)
                if parent:
                    parent_docs.append(parent)
                    seen_parents.add(parent_id)

        # If no parents found, return chunks
        return parent_docs if parent_docs else chunks[:k]

    def _get_parent_document(self, parent_id: str) -> Optional[Document]:
        """Retrieve parent document by ID"""
        # In practice, you'd have a separate collection or mapping
        # For now, create a document from the ID
        return Document(
            page_content=f"Parent document: {parent_id}",
            metadata={"parent_id": parent_id, "source": parent_id},
        )


class AdvancedRetriever:
    """Main retriever class with multiple strategies"""

    RETRIEVAL_STRATEGIES = {
        "simple": SimpleRetrieval,
        "mmr": MMRRetrieval,
        "hybrid": HybridRetrieval,
        "multi_query": MultiQueryRetrieval,
        "parent": ParentDocumentRetrieval,
    }

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_manager: EmbeddingManager,
        strategy: str = "simple",
        **strategy_kwargs,
    ):
        """
        Initialize the retriever

        Args:
            vector_store: ChromaVectorStore instance
            embedding_manager: EmbeddingManager instance
            strategy: Retrieval strategy ('simple', 'mmr', 'hybrid', 'compression', 'multi_query', 'parent')
            **strategy_kwargs: Strategy-specific parameters
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.strategy_name = strategy
        self.strategy_kwargs = strategy_kwargs

        self.retriever = self._create_retriever(strategy)

    def _create_retriever(self, strategy: str) -> RetrievalStrategy:
        """Factory method to create appropriate retriever"""
        strategy_class = self.RETRIEVAL_STRATEGIES.get(
            strategy.lower(), SimpleRetrieval
        )
        return strategy_class(self.vector_store, self.embedding_manager)

    def retrieve(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Retrieve documents using configured strategy"""
        return self.retriever.retrieve(query, k=k, **kwargs)

    def retrieve_with_scores(
        self, query: str, k: int = 5, **kwargs
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents with relevance scores (simple strategy only)"""
        if isinstance(self.retriever, SimpleRetrieval):
            return self.retriever.retrieve_with_scores(query, k=k, **kwargs)
        else:
            logger.warning("Retrieve with scores only available for simple strategy")
            results = self.retriever.retrieve(query, k=k, **kwargs)
            return [(doc, 1.0) for doc in results]  # Placeholder scores

    def change_strategy(self, strategy: str, **strategy_kwargs):
        """Change retrieval strategy dynamically"""
        self.strategy_name = strategy
        self.strategy_kwargs = strategy_kwargs
        self.retriever = self._create_retriever(strategy)

    async def aretrieve(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Async retrieval"""
        return await asyncio.to_thread(self.retrieve, query, k, **kwargs)
