from ingestion.pipeline import IngestionPipeline
from ingestion.config import config
import json


def test_retrieval():
    """Test retrieval after ingestion"""

    print("Testing Document Retrieval")
    print("=" * 60)

    # Run pipeline first
    pipeline = IngestionPipeline(config)

    # Try different queries
    queries = [
        "What is Artificial Intelligence?",
        # "What is machine learning?",
        # "Tell me about data science",
        # "Summary of the document",
        # "Key findings",
        # "What are the main topics?",
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 40)

        results = pipeline.vector_store.search(query, k=3)

        if not results:
            print("   ❌ No results found")
            continue

        for i, doc in enumerate(results, 1):
            # Truncate content for display
            content = (
                doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content
            )
            print(f"\n   Result {i}:")
            print(f"   📄 Content: {content}")
            print(f"   📁 Source: {doc.metadata.get('source', 'unknown')}")
            print(f"   🏷️  Section: {doc.metadata.get('section_title', 'N/A')}")

            # Show similarity score if available
            if "score" in doc.metadata:
                print(f"   📊 Score: {doc.metadata['score']:.4f}")


if __name__ == "__main__":
    test_retrieval()
