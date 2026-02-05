"""
Example: Using the Extended LLM Reranker with Query Decomposition and Result Aggregation

This example demonstrates the extended reranking pipeline:
1. Pre-ranking: Decompose complex queries into sub-queries
2. Reranking: Apply LLM reranking to each sub-query
3. Post-ranking: Aggregate results using coverage-based methods
"""
import sys
sys.path.insert(0, 'src')

# Basic imports
from autollmrerank.utils import Result
from autollmrerank.query_decomposer import (
    PassThroughDecomposer,
    DecomposedQuery,
)
from autollmrerank.result_aggregator import (
    PassThroughAggregator,
    RRFAggregator,
    CoverageAggregator,
    MMRAggregator,
)


def create_sample_results():
    """Create sample results for demonstration."""
    # Simulate results from different sub-queries
    hits1 = [
        {'docid': 'd1', 'score': 1.0, 'rank': 1, 'content_dict': {'text': 'ML basics'}},
        {'docid': 'd2', 'score': 0.8, 'rank': 2, 'content_dict': {'text': 'Deep learning'}},
        {'docid': 'd3', 'score': 0.6, 'rank': 3, 'content_dict': {'text': 'Neural networks'}},
    ]
    
    hits2 = [
        {'docid': 'd2', 'score': 1.0, 'rank': 1, 'content_dict': {'text': 'Deep learning'}},
        {'docid': 'd4', 'score': 0.9, 'rank': 2, 'content_dict': {'text': 'AI applications'}},
        {'docid': 'd1', 'score': 0.5, 'rank': 3, 'content_dict': {'text': 'ML basics'}},
    ]
    
    result1 = Result(qid='q1', query='machine learning basics', hits=hits1)
    result2 = Result(qid='q1', query='AI applications', hits=hits2)
    
    return [result1, result2]


def demo_passthrough_decomposer():
    """Demonstrate PassThroughDecomposer (no decomposition)."""
    print("=" * 60)
    print("PassThroughDecomposer Demo")
    print("=" * 60)
    
    decomposer = PassThroughDecomposer()
    query = "What is machine learning and its applications?"
    
    decomposed = decomposer.decompose(query)
    
    print(f"Original query: {decomposed.original_query}")
    print(f"Sub-queries: {decomposed.sub_queries}")
    print(f"Weights: {decomposed.weights}")
    print()


def demo_rrf_aggregator():
    """Demonstrate Reciprocal Rank Fusion aggregation."""
    print("=" * 60)
    print("RRFAggregator Demo (Reciprocal Rank Fusion)")
    print("=" * 60)
    
    results = create_sample_results()
    aggregator = RRFAggregator(k=60)
    
    aggregated = aggregator.aggregate(
        sub_query_results=results,
        original_query="What is ML and its applications?",
        qid='q1'
    )
    
    print(f"Query: {aggregated.query}")
    print("Aggregated ranking:")
    for hit in aggregated.hits:
        print(f"  Rank {hit['rank']}: {hit['docid']} (score: {hit['score']:.4f})")
    print()


def demo_coverage_aggregator():
    """Demonstrate Coverage-based aggregation."""
    print("=" * 60)
    print("CoverageAggregator Demo")
    print("=" * 60)
    
    results = create_sample_results()
    aggregator = CoverageAggregator(coverage_weight=0.5)
    
    aggregated = aggregator.aggregate(
        sub_query_results=results,
        weights=[0.5, 0.5],
        original_query="What is ML and its applications?",
        qid='q1'
    )
    
    print(f"Query: {aggregated.query}")
    print("Coverage-based ranking:")
    for hit in aggregated.hits:
        print(f"  Rank {hit['rank']}: {hit['docid']} (score: {hit['score']:.4f})")
    print()


def demo_mmr_aggregator():
    """Demonstrate MMR (Maximal Marginal Relevance) aggregation."""
    print("=" * 60)
    print("MMRAggregator Demo (Maximal Marginal Relevance)")
    print("=" * 60)
    
    results = create_sample_results()
    aggregator = MMRAggregator(lambda_param=0.7)
    
    aggregated = aggregator.aggregate(
        sub_query_results=results,
        original_query="What is ML and its applications?",
        qid='q1'
    )
    
    print(f"Query: {aggregated.query}")
    print("MMR-based ranking:")
    for hit in aggregated.hits:
        print(f"  Rank {hit['rank']}: {hit['docid']} (score: {hit['score']:.4f})")
    print()


def demo_full_pipeline():
    """Demonstrate the full pipeline with decomposition and aggregation."""
    print("=" * 60)
    print("Full Pipeline Demo: Decomposition -> Reranking -> Aggregation")
    print("=" * 60)
    
    # Step 1: Query Decomposition
    decomposer = PassThroughDecomposer()
    query = "What is machine learning and what are its practical applications?"
    decomposed = decomposer.decompose(query)
    
    print(f"Original query: {query}")
    print(f"Decomposed into {len(decomposed.sub_queries)} sub-queries:")
    for i, sq in enumerate(decomposed.sub_queries):
        print(f"  {i+1}. {sq} (weight: {decomposed.weights[i]:.2f})")
    
    # Step 2: Simulate reranking for each sub-query
    # (In practice, this would call the actual reranker)
    print("\nSimulated reranking for each sub-query...")
    results = create_sample_results()
    
    # Step 3: Result Aggregation
    aggregator = CoverageAggregator(coverage_weight=0.5)
    final_result = aggregator.aggregate(
        sub_query_results=results,
        weights=decomposed.weights,
        original_query=decomposed.original_query,
        qid='q1'
    )
    
    print("\nFinal aggregated ranking:")
    for hit in final_result.hits:
        print(f"  Rank {hit['rank']}: {hit['docid']} (score: {hit['score']:.4f})")
    print()


if __name__ == "__main__":
    demo_passthrough_decomposer()
    demo_rrf_aggregator()
    demo_coverage_aggregator()
    demo_mmr_aggregator()
    demo_full_pipeline()
    
    print("=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
