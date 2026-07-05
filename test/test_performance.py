from ingestion.pipeline import IngestionPipeline
from ingestion.config import config
import time
import os


def test_performance():
    """Benchmark pipeline performance"""

    print("Performance Benchmark")
    print("=" * 60)

    # Ensure documents exist
    if not os.path.exists(config.input_dir):
        print(f"❌ No input directory found at {config.input_dir}")
        return

    files = list(os.listdir(config.input_dir))
    if not files:
        print("❌ No files to process")
        return

    print(f"📄 Processing {len(files)} files...")
    print(f"📊 Chunk size: {config.chunk_size}")
    print(f"🔢 Embedding model: {config.embedding_model}")

    # Time the pipeline
    start = time.time()

    pipeline = IngestionPipeline(config)
    result = pipeline.process()

    end = time.time()
    total_time = end - start

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS:")
    print("=" * 60)

    if result["status"] == "success":
        print(f"✅ Pipeline completed in {total_time:.2f} seconds")
        print(f"📄 Documents processed: {result['documents_loaded']}")
        print(f"✂️  Chunks created: {result['chunks_created']}")
        print(f"⚡ Speed: {result['chunks_created']/total_time:.2f} chunks/second")
        print(f"💾 Documents in VDB: {result['collection_stats']['document_count']}")
        print(f"🔢 Embedding dimension: {result.get('embedding_dimension', 'unknown')}")
    else:
        print(f"❌ Pipeline failed: {result.get('reason')}")


if __name__ == "__main__":
    test_performance()
