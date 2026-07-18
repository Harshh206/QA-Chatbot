# QA Chatbot Documentation

## 1. Project Overview

This project is a Retrieval-Augmented Generation (RAG) based question answering system. It ingests documents from a local folder or a single file, splits them into smaller chunks, creates embeddings, stores them in ChromaDB, and uses an LLM to answer user questions based on the retrieved context.

The system is designed to work with local models through Ollama and supports:

- Document ingestion from multiple file formats
- Text preprocessing and metadata extraction
- Chunking and embedding generation
- Semantic retrieval from ChromaDB
- Optional reranking strategies
- Interactive chat mode

---

## 2. High-Level Architecture

The application is organized into three main stages:

1. Ingestion
   - Load documents
   - Clean and enrich them
   - Split them into chunks
   - Generate embeddings and store them

2. Retrieval
   - Process the user query
   - Search the vector database
   - Optionally rerank the results

3. Generation
   - Build context from retrieved documents
   - Send the prompt to the LLM
   - Return the final answer with sources

---

## 3. Main Entry Points

### main.py
The main script is the primary command-line interface for the project.

It allows you to:

- Ingest files or entire directories
- Run a single query
- Start interactive chat mode
- Choose retrieval strategy and reranker

### chat.py
This module provides an interactive chat interface built on top of the RAG pipeline.

### rag_chain.py
This file combines retrieval and generation into a single RAG workflow.

---

## 4. Step-by-Step Workflow

### Step 1: Configuration

The configuration is defined in [config.py](../config.py).

The main settings include:

- Chunk size and chunk overlap
- LLM model name and base URL
- Embedding model name
- ChromaDB storage directory
- Input directory for documents

The central object is `PipelineConfig`, which stores all configurable parameters.

### Step 2: Document Loading

The document loading phase is handled by [ingestion/loader.py](../ingestion/loader.py).

Key responsibilities:

- Detect the file type from its extension
- Use the correct loader for PDF, DOCX, Markdown, TXT, CSV, or JSON files
- Add metadata such as file name, file type, source path, and size

Important function:

- `DocumentLoader.load_single_file(file_path)`: loads one file
- `DocumentLoader.load_directory()`: loads all supported files from a directory

### Step 3: Markdown Conversion

The conversion step is handled by [ingestion/processors/markdown.py](../ingestion/processors/markdown.py).

This module converts supported document formats to Markdown when possible so that structure and headings are preserved better.

Important function:

- `MarkdownConverter.convert_document(document)`: converts a document to Markdown
- `MarkdownConverter.process_documents(documents)`: processes a list of documents

### Step 4: Text Cleaning

The cleaning stage is managed by [ingestion/processors/cleaner.py](../ingestion/processors/cleaner.py).

This step removes excessive whitespace, weird Unicode artifacts, and other formatting noise from the document text.

Important function:

- `TextCleaner.clean_document(document)`: cleans one document
- `TextCleaner.batch_clean(documents)`: cleans a list of documents

### Step 5: Metadata Extraction

Metadata extraction is handled by [ingestion/processors/metadata_extract.py](../ingestion/processors/metadata_extract.py).

This step enriches each document with metadata such as:

- Title
- Language
- Word count
- Character count
- Keywords
- Content hash

Important function:

- `MetadataExtractor.extract_metadata(document)`: enriches one document
- `MetadataExtractor.batch_extract(documents)`: processes many documents

### Step 6: Chunking

Chunking is implemented in [ingestion/chunker.py](../ingestion/chunker.py).

The system splits documents into smaller pieces so that retrieval remains efficient and focused.

Supported strategies:

- Recursive chunking
- Markdown-aware chunking
- Hybrid chunking
- Token-based chunking

Important function:

- `ChunkStrategy.split(documents)`: base method for all chunkers
- `get_chunker(strategy, **kwargs)`: creates the requested chunker

### Step 7: Embedding Generation

Embeddings are created by [ingestion/embedding.py](../ingestion/embedding.py).

This module uses an embedding model from Ollama to transform text into vectors that can be compared semantically.

Important function:

- `EmbeddingManager.embed_documents(texts)`: creates embeddings for many texts
- `EmbeddingManager.embed_query(query)`: creates an embedding for a single query
- `EmbeddingManager.embed_chunks(chunks)`: attaches embeddings to document chunks

### Step 8: Vector Storage in ChromaDB

Vector storage is managed by [ingestion/vectorstore.py](../ingestion/vectorstore.py).

Documents and chunks are stored in ChromaDB so the system can retrieve relevant content later using similarity search.

Important function:

- `ChromaVectorStore.add_documents(documents)`: stores documents
- `ChromaVectorStore.search(query, k)`: retrieves similar documents
- `ChromaVectorStore.get_collection_stats()`: returns storage statistics

### Step 9: Ingestion Pipeline Execution

The complete ingestion process is orchestrated by [ingestion/pipeline.py](../ingestion/pipeline.py).

The main class is `IngestionPipeline`.

Important function:

- `IngestionPipeline.process()`: runs the full ingestion workflow
- `IngestionPipeline.process_single_file(file_path)`: processes one file
- `IngestionPipeline.process_directory(dir_path)`: processes a folder

### Step 10: Query Processing

Before retrieval, the query is normalized by [retrieval/query_processor.py](../retrieval/query_processor.py).

This module:

- Lowercases the query
- Removes extra whitespace
- Strips unnecessary characters
- Can expand or enhance the query for better retrieval

Important function:

- `QueryProcessor.preprocess_query(query)`: cleans the query
- `QueryProcessor.expand_query(query)`: creates related query variants
- `QueryProcessor.enhance_query(query)`: produces an enriched query profile

### Step 11: Retrieval

Retrieval is implemented in [retrieval/retriever.py](../retrieval/retriever.py).

The system supports different retrieval strategies:

- `simple`: basic vector similarity search
- `mmr`: Maximum Marginal Relevance for diverse results
- `hybrid`: combines semantic and keyword search
- `multi_query`: searches multiple query variations
- `parent`: retrieves parent documents based on chunks

Important function:

- `AdvancedRetriever.retrieve(query, k)`: runs retrieval using the selected strategy
- `AdvancedRetriever.change_strategy(strategy)`: switches strategies dynamically

### Step 12: Reranking

Reranking is handled by [retrieval/reranker.py](../retrieval/reranker.py).

This step reorders retrieved documents to improve relevance.

Supported rerankers:

- `cross_encoder`: uses a cross-encoder model
- `llm`: uses an LLM to score documents
- `rrf`: uses reciprocal rank fusion
- `ensemble`: combines several rerankers

Important function:

- `CrossEncoderReranker.rerank(query, documents)`: reranks with a cross-encoder
- `LLMReranker.rerank(query, documents)`: reranks using an LLM
- `RRFReranker.rerank(query, documents)`: reranks based on ranking scores

### Step 13: Retrieval Pipeline Orchestration

The orchestration layer is [retrieval/retrieval_pipeline.py](../retrieval/retrieval_pipeline.py).

The `RetrievalPipeline` class combines:

- Query processing
- Document retrieval
- Optional reranking
- Metadata formatting

Important function:

- `RetrievalPipeline.retrieve(query, k)`: main retrieval method
- `RetrievalPipeline.retrieve_with_metadata(query, k)`: returns metadata-rich results
- `RetrievalPipeline.retrieve_context(query, k)`: prepares a formatted context block

### Step 14: LLM Answer Generation

The LLM interface is implemented in [llm.py](../llm.py).

This module communicates with Ollama using the `ChatOllama` model.

Important function:

- `LLMManager.generate(prompt, system_prompt)`: generates a response
- `LLMManager.generate_with_context(query, context)`: creates an answer from retrieved context
- `LLMManager.stream_generate(prompt, system_prompt)`: streams the LLM output

### Step 15: RAG Chain Execution

The RAG workflow is implemented in [rag_chain.py](../rag_chain.py).

The class `RAGChain` connects retrieval and generation:

1. Retrieve documents for the query
2. Format the retrieved context
3. Send the context and query to the LLM
4. Return the answer and sources

Important function:

- `RAGChain.ask(query)`: performs a full question-answer cycle
- `RAGChain.stream_ask(query)`: streams the answer token by token
- `RAGChain._format_context(documents)`: formats retrieved documents into prompt context
- `RAGChain._format_sources(documents)`: prepares document source metadata

### Step 16: Interactive Chat Mode

The chat experience is provided by [chat.py](../chat.py).

The `ChatSession` stores conversation history, while `run_chat()` begins an interactive loop.

Important function:

- `ChatSession.ask(question)`: sends one question through the RAG pipeline
- `ChatSession.show_history(limit)`: displays recent conversation turns
- `ChatSession.clear()`: clears the history
- `run_chat(...)`: starts the chat loop

---

## 5. Module-by-Module Summary

### [config.py](../config.py)
Contains global configuration values for the entire project.

### [main.py](../main.py)
CLI entry point for ingestion and querying.

### [chat.py](../chat.py)
Interactive chat interface.

### [llm.py](../llm.py)
Wraps the Ollama LLM integration.

### [rag_chain.py](../rag_chain.py)
Combines retrieval and generation into a single RAG chain.

### [ingestion/loader.py](../ingestion/loader.py)
Loads documents from supported file types.

### [ingestion/pipeline.py](../ingestion/pipeline.py)
Coordinates the ingestion workflow.

### [ingestion/chunker.py](../ingestion/chunker.py)
Splits documents into chunks.

### [ingestion/embedding.py](../ingestion/embedding.py)
Generates embeddings for chunks and queries.

### [ingestion/vectorstore.py](../ingestion/vectorstore.py)
Stores and searches embeddings in ChromaDB.

### [retrieval/query_processor.py](../retrieval/query_processor.py)
Preprocesses and enhances the incoming user query.

### [retrieval/retriever.py](../retrieval/retriever.py)
Implements different retrieval strategies.

### [retrieval/reranker.py](../retrieval/reranker.py)
Reorders retrieved documents by relevance.

### [retrieval/retrieval_pipeline.py](../retrieval/retrieval_pipeline.py)
Orchestrates retrieval and reranking.

---

## 6. How to Run the Project

### Install dependencies

Install the required packages from [requriement.txt](../requriement.txt).

### Start Ollama
Make sure your local Ollama server is running and that the configured model is available.

### Ingest documents

Example:

```bash
python main.py --dir ./data
```

### Ask a single question

Example:

```bash
python main.py --query "What is this project about?" --strategy hybrid --k 3
```

### Start interactive chat

Example:

```bash
python main.py --chat
```

---

## 7. Typical Data Flow

A typical request flows through the system like this:

1. User provides a question
2. The query is cleaned and processed
3. The retrieval pipeline searches ChromaDB
4. Relevant chunks are reranked if enabled
5. The RAG chain builds a context block
6. The LLM answers using that context
7. The response is returned with sources

---

## 8. Notes and Extension Points

This project is a solid foundation for a local RAG chatbot. You can extend it by:

- Adding more document formats
- Using a stronger reranker
- Switching to a different embedding model
- Adding caching or hybrid search improvements
- Adding support for conversational memory beyond simple history

---

## 9. Summary

In short, this project works as follows:

- It reads documents
- Converts and cleans them
- Splits them into chunks
- Embeds and stores them
- Retrieves relevant chunks for a user question
- Uses an LLM to generate an answer from the retrieved context

That is the core idea behind the QA chatbot.
