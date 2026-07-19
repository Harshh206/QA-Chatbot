from ingestion.pipeline import IngestionPipeline
from config import config
import json
import os


def test_pipeline():
    """Test the complete pipeline with sample documents"""

    print("=" * 60)
    print("Testing Document Ingestion Pipeline")
    print("=" * 60)

    # 1. Check Ollama connection
    print("\n[1] Checking Ollama connection...")
    from ingestion.embedding import EmbeddingManager

    test_manager = EmbeddingManager(
        model_name=config.embedding_model, base_url=config.base_url
    )

    if test_manager.validate_connection():
        print(f"✅ Connected to Ollama at {config.base_url}")
        print(f"   Model: {config.embedding_model}")
        dim = test_manager.get_embedding_dimension()
        if dim:
            print(f"   Embedding dimension: {dim}")
    else:
        print(f"❌ Cannot connect to Ollama. Please run: ollama serve")
        print(f"   Also ensure model is pulled: ollama pull {config.embedding_model}")
        return

    # 2. Check if there are documents to process
    print("\n[2] Checking for input documents...")
    if not os.path.exists(config.input_dir):
        print(f"❌ Input directory not found: {config.input_dir}")
        print("   Create the directory and add some documents")
        return

    files = [
        f
        for f in os.listdir(config.input_dir)
        if os.path.isfile(os.path.join(config.input_dir, f))
    ]

    if not files:
        print(f"⚠️  No files found in {config.input_dir}")
        print("   Please add some documents to process")
        return

    print(f"✅ Found {len(files)} files: {', '.join(files[:5])}")

    # 3. Run the pipeline
    print("\n[3] Running ingestion pipeline...")
    pipeline = IngestionPipeline(config)
    result = pipeline.process()

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)

    if result["status"] == "success":
        print(f"✅ Pipeline completed successfully")
        print(f"   📄 Documents loaded: {result['documents_loaded']}")
        print(f"   ✂️  Chunks created: {result['chunks_created']}")
        print(
            f"   💾 Stored in ChromaDB: {result['collection_stats']['document_count']}"
        )
        print(
            f"   🔢 Embedding dimension: {dim or 'unknown'}"
        )
        print(f"   🆔 Sample IDs: {', '.join(result['stored_ids'][:3])}")
    else:
        print(f"❌ Pipeline failed: {result.get('reason', 'Unknown error')}")

    return result


if __name__ == "__main__":
    test_pipeline()
