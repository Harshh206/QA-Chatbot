import subprocess
import sys
import os
from pathlib import Path


def run_all_tests():
    """Run all tests sequentially"""
    repo_root = Path(__file__).resolve().parents[1]
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    python_executable = str(venv_python) if venv_python.exists() else sys.executable

    print("\n" + "=" * 70)
    print(" RUNNING COMPLETE TEST SUITE")
    print("=" * 70)

    def run_test(module_name: str):
        print("\n" + "=" * 70)
        print(f"[Running] {module_name}")
        print("=" * 70)
        subprocess.run(
            [python_executable, "-m", module_name],
            cwd=str(repo_root),
            env={**dict(os.environ), "PYTHONPATH": str(repo_root)},
            check=False,
        )

    run_test("test.test_embeddings")
    run_test("test.test_vectorstore")
    run_test("test.test_retrieval")
    run_test("test.test_pipeline")
    run_test("test.test_performance")


if __name__ == "__main__":
    run_all_tests()
