from ingestion.embeddings import EmbeddingManager
from ingestion.vector_store import ChromaVectorStore
from langchain_core.documents import Document
from config import config

def test_vector_store():
    """Test vector store operations"""

    print("Testing Vector Store (ChromaDB)")
    print("-" * 40)

    # Initialize
    embedding_manager = EmbeddingManager(
        model_name=config.embedding_model, base_url=config.base_url
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
            page_content="Artificial Intelligence is today widely used for various applications like computer vision, speech recognition, decision-making, perception, reasoning, cognitive capabilities, and so on.",
            metadata={"source": "test1", "category": "ai"},
        ),
        Document(
            page_content="Artificial Neural Network (ANN) is a computational model based on the structure of the Biological Neural Network(BNN). The human brain has billions of neurons that collect, process the information, and drive meaningful results out of it. The neurons use electro-chemical signals to communicate and pass the information to other neurons. Similarly, ANN consists of artificial neurons called nodes connected with other nodes forming a complex relationship between the output and the input.",
            metadata={"source": "test2", "category": "ann"},
        ),
    ]

    ids = vector_store.add_documents(test_docs)
    print(f"   ✅ Added {len(ids)} test documents")
    print(f"   🆔 IDs: {ids[:3]}")

    # 3. Search
    print("\n2. Test documents search:")
    print("\n3. 'What is Artificial Intelligence?':")
    results = vector_store.search("What is Artificial Intelligence?", k=2)

    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc.page_content}")
        print(f"      Source: {doc.metadata.get('source', 'unknown')}")
        print(f"      Score: {doc.metadata.get('score', 'N/A')}")

    # 4. Search with scores
    print("\n4. Search with similarity scores of 'Artificial Intelligence':")
    results_with_scores = vector_store.search_with_score("What is Artificial Intelligence?", k=1)

    for doc, score in results_with_scores:
        print(f"   📝 {doc.page_content[:50]}...")
        print(f"   📊 Score: {score:.4f}")

if __name__ == "__main__":
    test_vector_store()
