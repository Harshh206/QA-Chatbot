import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    # Chunking settings
    chunk_size: int = 768
    chunk_overlap: int = 100

    # Embedding settings
    embedding_model: str = "qwen3-embedding:0.6b"
    ollama_base_url: str = "http://localhost:11434"
    embedding_dimension: int = 512

    # Vector DB settings
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "my_rag_collection"

    # File paths
    input_dir: str = "./data"

    # Logging
    log_level: str = "INFO"


config = PipelineConfig()
