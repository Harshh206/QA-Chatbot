from ingestion.embedding import EmbeddingManager
from ingestion.vectorstore import ChromaVectorStore
from langchain_core.documents import Document
from ingestion.config import config

def test_vector_store():
    """Test vector store operations"""

    print("Testing Vector Store (ChromaDB)")
    print("-" * 40)

    # Initialize
    embedding_manager = EmbeddingManager(
        model_name=config.embedding_model, base_url=config.ollama_base_url
    )

    vector_store = ChromaVectorStore(
        embedding_manager=embedding_manager,
        persist_directory=config.chroma_persist_dir,
        collection_name=config.collection_name,
    )

    # 1. Check collection stats
    print("\n1. Collection statistics:")
    stats = vector_store.get_collection_stats()
    print(f"   📚 Collection: {stats.get('collection_name')}")
    print(f"   📄 Document count: {stats.get('document_count', 0)}")
    print(f"   💾 Persist directory: {stats.get('persist_directory')}")

    # 2. Add test documents
    print("\n2. Adding test documents:")
    test_docs = [
        Document(
            page_content="Python is a programming language.",
            metadata={"source": "test1", "category": "programming"},
        ),
        Document(
            page_content="Machine learning is a subset of AI.",
            metadata={"source": "test2", "category": "ai"},
        ),
        Document(
            page_content="Data science involves statistics and programming.",
            metadata={"source": "test3", "category": "data"},
        ),
    ]

    ids = vector_store.add_documents(test_docs)
    print(f"   ✅ Added {len(ids)} test documents")
    print(f"   🆔 IDs: {ids[:3]}")

    # 3. Search
    print("\n3. Searching for 'programming':")
    results = vector_store.search("programming", k=2)

    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc.page_content}")
        print(f"      Source: {doc.metadata.get('source', 'unknown')}")
        print(f"      Score: {doc.metadata.get('score', 'N/A')}")

    # 4. Search with scores
    print("\n4. Search with similarity scores:")
    results_with_scores = vector_store.search_with_score("AI", k=2)

    for doc, score in results_with_scores:
        print(f"   📝 {doc.page_content[:50]}...")
        print(f"   📊 Score: {score:.4f}")

    # # 5. Clean up test documents
    # print("\n5. Cleaning up test documents...")
    # vector_store.delete_collection()
    # print("   ✅ Test collection deleted")


if __name__ == "__main__":
    test_vector_store()
