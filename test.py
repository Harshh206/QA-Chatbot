#!/usr/bin/env python
# test_rag.py

import logging
from config import config
from rag_chain import create_rag_system, RAGEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_rag():
    """Test basic RAG functionality"""
    logger.info("Testing basic RAG...")
    
    rag_system = create_rag_system(
        config,
        retriever_strategy='hybrid',
        reranker_type='cross_encoder',
        model_name='qwen2.5:7b',
        include_sources=True,
        k=3
    )
    
    # Test queries
    queries = [
        "What is the main topic of the document?",
        "Can you summarize the key points?",
    ]
    
    for query in queries:
        logger.info(f"\nQuery: {query}")
        result = rag_system.ask(query)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {len(result['sources'])}")
        print(f"Documents used: {result['document_count']}")
        print("-" * 40)


def test_advanced_rag():
    """Test advanced RAG features"""
    logger.info("Testing advanced RAG...")
    
    rag_system = create_rag_system(
        config,
        retriever_strategy='hybrid',
        reranker_type='cross_encoder',
        model_name='qwen2.5:7b',
        include_sources=True,
        k=5
    )
    
    # Test with history
    print("\nTesting with conversation history:")
    result1 = rag_system.ask_with_history("What is the main topic?")
    print(f"Q1: {result1['answer'][:100]}...")
    
    result2 = rag_system.ask_with_history("Can you elaborate on that?")
    print(f"Q2: {result2['answer'][:100]}...")
    
    # Test decomposition
    if hasattr(rag_system, 'ask_with_decomposition'):
        print("\nTesting query decomposition:")
        complex_query = "What are the main topics and how are they related?"
        result = rag_system.ask_with_decomposition(complex_query)
        print(f"Complex query answer: {result['answer'][:150]}...")
        
        if 'sub_queries' in result:
            print(f"Sub-queries: {result['sub_queries']}")


def test_evaluation():
    """Test RAG evaluation"""
    logger.info("Testing RAG evaluation...")
    
    rag_system = create_rag_system(
        config,
        retriever_strategy='simple',
        model_name= ,
        include_sources=False,
        k=3
    )
    
    evaluator = RAGEvaluator(rag_system)
    
    test_questions = [
        {'query': 'What is the main topic?'},
        {'query': 'Explain the key concepts.'},
        {'query': 'Summarize the document.'},
    ]
    
    results = evaluator.evaluate(test_questions)
    
    print("\nEvaluation Results:")
    print(f"Total questions: {results['total_questions']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Avg documents retrieved: {results['avg_documents_retrieved']:.2f}")
    
    # Print individual results
    for r in evaluator.evaluation_results:
        print(f"\nQuery: {r['query']}")
        if 'error' in r:
            print(f"Error: {r['error']}")
        else:
            print(f"Answer: {r['answer'][:100]}...")
            print(f"Documents: {r['document_count']}")


# def interactive_rag():
#     """Interactive RAG session"""
#     rag_system = create