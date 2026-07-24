from ingestion.embeddings import EmbeddingManager
from config import config
import time


def test_embeddings():
    """Test embedding functionality separately"""

    print("Testing Ollama Embeddings")
    print("-" * 40)

    # Initialize embedding manager
    manager = EmbeddingManager(
        model_name=config.embedding_model, base_url=config.base_url,
        dimensions=config.embedding_dimension
    )

    # Test connection
    print(f"\n1. Testing connection to {config.base_url}")
    if manager.validate_connection():
        print("   ✅ Connection successful")
    else:
        print("   ❌ Connection failed")
        return

    # Test single query embedding
    print(f"\n2. Testing single query embedding")
    test_query = "What is machine learning?"
    start = time.time()
    embedding = manager.embed_query(test_query)
    end = time.time()

    print(f"   ✅ Query: '{test_query}'")
    print(f"   📐 Vector dimension: {len(embedding)}")
    print(f"   ⏱️  Time: {(end - start)*1000:.2f}ms")
    print(f"   📊 First 5 values: {embedding[:5]}")

    # Test batch embedding
    print(f"\n3. Testing batch document embedding")
    test_docs = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by the human brain.",
        "Deep learning uses multiple layers of neural networks.",
    ]

    start = time.time()
    embeddings = manager.embed_documents(test_docs)
    end = time.time()

    print(f"   ✅ Embedded {len(test_docs)} documents")
    print(f"   📐 Vector dimension: {len(embeddings[0])}")
    print(f"   ⏱️  Time: {(end - start)*1000:.2f}ms")
    print(f"   📊 Similarity between doc 1 and 2:")

    # Calculate cosine similarity
    import numpy as np

    cos_sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    print(f"   🔢 Cosine similarity: {cos_sim:.4f}")


if __name__ == "__main__":
    test_embeddings()
