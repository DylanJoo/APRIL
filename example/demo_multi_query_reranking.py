"""
Example: Multi-Query Reranking with Result Aggregation

This example demonstrates the simplified multi-query reranking:
1. Attach sub-queries to a Result (externally generated)
2. Rerank with each sub-query using any base strategy
3. Aggregate results using RRF, Coverage, or MMR methods

Note: Complex query decomposition should be done externally (not in this framework).
This framework focuses on the reranking and aggregation logic.
"""
import sys
import os

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), 'src')
sys.path.insert(0, src_dir)

from autollmrerank.utils import Result
from autollmrerank.result_aggregator import (
    RRFAggregator,
    CoverageAggregator,
    MMRAggregator,
)


def create_sample_results():
    """Create sample reranked results from different sub-queries."""
    # Simulate reranked results for sub-query 1: "machine learning basics"
    hits1 = [
        {'docid': 'd1', 'score': 1.0, 'rank': 1, 'content_dict': {'text': 'ML basics'}},
        {'docid': 'd2', 'score': 0.8, 'rank': 2, 'content_dict': {'text': 'Deep learning'}},
        {'docid': 'd3', 'score': 0.6, 'rank': 3, 'content_dict': {'text': 'Neural networks'}},
    ]
    
    # Simulate reranked results for sub-query 2: "AI applications"
    hits2 = [
        {'docid': 'd2', 'score': 1.0, 'rank': 1, 'content_dict': {'text': 'Deep learning'}},
        {'docid': 'd4', 'score': 0.9, 'rank': 2, 'content_dict': {'text': 'AI applications'}},
        {'docid': 'd1', 'score': 0.5, 'rank': 3, 'content_dict': {'text': 'ML basics'}},
    ]
    
    result1 = Result(qid='q1', query='machine learning basics', hits=hits1)
    result2 = Result(qid='q1', query='AI applications', hits=hits2)
    
    return [result1, result2]


def demo_rrf_aggregation():
    """Demonstrate Reciprocal Rank Fusion aggregation."""
    print("=" * 60)
    print("RRF Aggregation Demo")
    print("=" * 60)
    
    sub_results = create_sample_results()
    aggregator = RRFAggregator(k=60)
    
    aggregated = aggregator.aggregate(
        sub_query_results=sub_results,
        original_query="What is ML and its applications?",
        qid='q1'
    )
    
    print(f"Query: {aggregated.query}")
    print("Aggregated ranking (RRF):")
    for hit in aggregated.hits:
        print(f"  Rank {hit['rank']}: {hit['docid']} (score: {hit['score']:.4f})")
    print()


def demo_coverage_aggregation():
    """Demonstrate Coverage-based aggregation."""
    print("=" * 60)
    print("Coverage Aggregation Demo")
    print("=" * 60)
    
    sub_results = create_sample_results()
    aggregator = CoverageAggregator(coverage_weight=0.5)
    
    aggregated = aggregator.aggregate(
        sub_query_results=sub_results,
        weights=[0.5, 0.5],
        original_query="What is ML and its applications?",
        qid='q1'
    )
    
    print(f"Query: {aggregated.query}")
    print("Coverage-based ranking:")
    for hit in aggregated.hits:
        print(f"  Rank {hit['rank']}: {hit['docid']} (score: {hit['score']:.4f})")
    print()


def demo_mmr_aggregation():
    """Demonstrate MMR (Maximal Marginal Relevance) aggregation."""
    print("=" * 60)
    print("MMR Aggregation Demo")
    print("=" * 60)
    
    sub_results = create_sample_results()
    aggregator = MMRAggregator(lambda_param=0.7)
    
    aggregated = aggregator.aggregate(
        sub_query_results=sub_results,
        original_query="What is ML and its applications?",
        qid='q1'
    )
    
    print(f"Query: {aggregated.query}")
    print("MMR-based ranking:")
    for hit in aggregated.hits:
        print(f"  Rank {hit['rank']}: {hit['docid']} (score: {hit['score']:.4f})")
    print()


def demo_multi_query_workflow():
    """
    Demonstrate the complete multi-query reranking workflow.
    
    In a real scenario:
    1. Sub-queries are generated externally (e.g., by a query decomposition service)
    2. Each sub-query is used to rerank the candidate documents
    3. Results from all sub-queries are aggregated
    """
    print("=" * 60)
    print("Complete Multi-Query Workflow Demo")
    print("=" * 60)
    
    # Step 1: Original query and sub-queries (externally generated)
    original_query = "What is machine learning and what are its practical applications?"
    sub_queries = [
        "What is machine learning?",
        "Machine learning applications in industry"
    ]
    weights = [0.6, 0.4]  # Weight the first aspect more heavily
    
    print(f"Original query: {original_query}")
    print(f"Sub-queries (externally generated):")
    for i, sq in enumerate(sub_queries):
        print(f"  {i+1}. {sq} (weight: {weights[i]:.2f})")
    
    # Step 2: Simulate reranking for each sub-query
    # In practice, this would use the actual reranker
    print("\nSimulating reranking for each sub-query...")
    sub_results = create_sample_results()
    
    # Step 3: Aggregate using RRF
    aggregator = RRFAggregator(k=60)
    final_result = aggregator.aggregate(
        sub_query_results=sub_results,
        weights=weights,
        original_query=original_query,
        qid='q1'
    )
    
    print("\nFinal aggregated ranking:")
    for hit in final_result.hits:
        print(f"  Rank {hit['rank']}: {hit['docid']} (score: {hit['score']:.4f})")
    print()


def show_usage_with_assembler():
    """Show how to use attach_sub_queries with MultiQueryAssembler."""
    print("=" * 60)
    print("Usage with MultiQueryAssembler (code example)")
    print("=" * 60)
    
    example_code = '''
# Attach sub-queries to a Result for multi-query reranking
from autollmrerank.input_assembler import attach_sub_queries, MultiQueryAssembler

# Create your result
result = Result(qid='q1', query='original query', hits=[...])

# Attach sub-queries (generated externally)
attach_sub_queries(
    result,
    sub_queries=['What is ML?', 'ML applications'],
    weights=[0.6, 0.4]  # optional weights
)

# Create the multi-query assembler with a base strategy
assembler = MultiQueryAssembler(
    config=config,
    prompt_builder=prompt_builder,
    llm_provider=llm,
    result_parser=result_parser,
    base_strategy=base_strategy,  # e.g., SlidingWindow for RankGPT
    aggregator=RRFAggregator(k=60),
)

# Rerank - will use each sub-query and aggregate results
reranked = assembler.run(init_results=[result], rank_start=0, rank_end=100)
'''
    print(example_code)


if __name__ == "__main__":
    demo_rrf_aggregation()
    demo_coverage_aggregation()
    demo_mmr_aggregation()
    demo_multi_query_workflow()
    show_usage_with_assembler()
    
    print("=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
