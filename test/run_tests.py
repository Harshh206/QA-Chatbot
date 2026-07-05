import subprocess
import sys
import os


def run_all_tests():
    """Run all tests sequentially"""

    print("\n" + "=" * 70)
    print(" RUNNING COMPLETE TEST SUITE")
    print("=" * 70)

    # Test embeddings
    print("\n" + "=" * 70)
    print("[Step 1] Testing Embeddings...")
    print("=" * 70)
    subprocess.run([sys.executable, "./test/test_embeddings.py"])

    # Test pipeline
    print("\n" + "=" * 70)
    print("[Step 2] Testing Pipeline...")
    print("=" * 70)
    subprocess.run([sys.executable, "./test/test_pipeline.py"])

    # Test retrieval
    print("\n" + "=" * 70)
    print("[Step 3] Testing Retrieval...")
    print("=" * 70)
    subprocess.run([sys.executable, "./test/test_retrieval.py"])

    # Performance test
    print("\n" + "=" * 70)
    print("[Step 4] Performance Benchmark...")
    print("=" * 70)
    subprocess.run([sys.executable, "./test/test_performance.py"])


if __name__ == "__main__":
    run_all_tests()
