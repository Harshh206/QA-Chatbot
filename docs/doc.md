[ User Query ]
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[ Dense Vector Search ]          [ Sparse BM25 Search ]
 (OpenAI / Cohere / BGE)          (Exact keywords/IDs)
       │                                 │
       └────────────────┬────────────────┘
                        ▼
           [ Combined Pool: Top 100 ]
                        │
                        ▼
             [ Cross-Encoder Reranker ]
            (e.g., Cohere v4 / BGE v2)
                        │
                        ▼
         [ Dynamic Score Filter > 0.70 ]
                        │
                        ▼
          [ Top 3-5 Hyper-Exact Chunks ] ───> To LLM