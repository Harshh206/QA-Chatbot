import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    # Chunking settings
    chunk_size: int = 640
    chunk_overlap: int = 96

    llm: str = "llama3:8b"

    # Embedding settings
    embedding_model: str = "qwen3-embedding:0.6b"
    ollama_base_url: str = "http://localhost:11434"
    embedding_dimension: int = 768
    # Vector DB settings
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "my_rag_collection"

    # File paths
    input_dir: str = "./data"

    # Logging
    log_level: str = "INFO"


config = PipelineConfig()
