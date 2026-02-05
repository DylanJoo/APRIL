"""
Unit tests for the MultiQueryAssembler logic.

These tests verify the multi-query reranking and aggregation logic
without requiring full dependencies.
"""
import sys
import os
import copy

# Add src directory to path dynamically based on script location
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), 'src')
sys.path.insert(0, src_dir)

# Import only lightweight modules
from autollmrerank.utils import Result
from autollmrerank.result_aggregator.base import RRFAggregator


def create_test_result(qid, query, hits_data):
    """Helper to create Result objects for testing."""
    hits = []
    for rank, (docid, score) in enumerate(hits_data, start=1):
        hits.append({
            'docid': docid,
            'score': score,
            'rank': rank,
            'content_dict': {'text': f'Document {docid} content'}
        })
    return Result(qid=qid, query=query, hits=hits)


def attach_sub_queries(result, sub_queries, weights=None):
    """
    Attach sub-queries to a Result object for multi-query reranking.
    
    Note: This is a local copy of the function from multi_query.py.
    We duplicate it here to avoid importing the full module which has
    heavy dependencies (transformers, etc.) that are not needed for unit tests.
    """
    result.sub_queries = sub_queries
    if weights is not None:
        result.sub_query_weights = weights
    else:
        result.sub_query_weights = [1.0 / len(sub_queries)] * len(sub_queries)
    return result


class MockRerankStrategy:
    """Mock reranking strategy for testing."""
    
    def __init__(self):
        self.call_count = 0
        self.last_queries = []
    
    def run(self, init_results, rank_start, rank_end, batch_size=8, num_runs=1, **kwargs):
        """Mock reranking - returns results with query-dependent ranking."""
        self.call_count += 1
        
        reranked = []
        for result in init_results:
            self.last_queries.append(result.query)
            
            # Simulate different rankings based on query
            new_hits = []
            for i, hit in enumerate(result.hits):
                new_hit = hit.copy()
                # Vary score based on query hash
                query_hash = hash(result.query) % 10
                new_hit['score'] = hit['score'] + (query_hash * 0.01)
                new_hits.append(new_hit)
            
            new_result = Result(
                qid=result.qid,
                query=result.query,
                hits=new_hits
            )
            reranked.append(new_result)
        
        return reranked


def _test_simulate_multi_query_rerank(result, base_strategy, aggregator):
    """
    Simulate the MultiQueryAssembler logic for testing.
    
    This mirrors the core logic of MultiQueryAssembler._rerank_multi_query()
    to allow testing without importing the full module and its dependencies.
    """
    sub_queries = getattr(result, 'sub_queries', None)
    
    if not sub_queries or len(sub_queries) <= 1:
        # Single query - use base strategy directly
        return base_strategy.run(init_results=[result], rank_start=0, rank_end=10)[0]
    
    # Multi-query - rerank with each sub-query
    weights = getattr(result, 'sub_query_weights', None)
    original_query = result.query
    sub_results = []
    
    for sub_query in sub_queries:
        sub_result = copy.deepcopy(result)
        sub_result.query = sub_query
        reranked = base_strategy.run(init_results=[sub_result], rank_start=0, rank_end=10)
        sub_results.extend(reranked)
    
    # Aggregate
    aggregated = aggregator.aggregate(
        sub_query_results=sub_results,
        weights=weights,
        original_query=original_query,
        qid=result.qid
    )
    return aggregated


class TestAttachSubQueries:
    """Tests for attach_sub_queries helper function."""
    
    def test_attach_with_weights(self):
        """Test attaching sub-queries with explicit weights."""
        result = create_test_result('q1', 'original query', [('d1', 1.0)])
        
        attach_sub_queries(result, ['sub1', 'sub2'], [0.6, 0.4])
        
        assert result.sub_queries == ['sub1', 'sub2']
        assert result.sub_query_weights == [0.6, 0.4]
    
    def test_attach_with_default_weights(self):
        """Test attaching sub-queries with default equal weights."""
        result = create_test_result('q1', 'original query', [('d1', 1.0)])
        
        attach_sub_queries(result, ['sub1', 'sub2', 'sub3'])
        
        assert result.sub_queries == ['sub1', 'sub2', 'sub3']
        assert len(result.sub_query_weights) == 3
        assert abs(sum(result.sub_query_weights) - 1.0) < 0.001


class TestMultiQueryLogic:
    """Tests for multi-query reranking logic."""
    
    def test_single_query_uses_base_strategy(self):
        """Test that single query (no sub-queries) uses base strategy directly."""
        mock_strategy = MockRerankStrategy()
        aggregator = RRFAggregator()
        
        result = create_test_result('q1', 'test query', [('d1', 1.0), ('d2', 0.5)])
        
        reranked = _test_simulate_multi_query_rerank(result, mock_strategy, aggregator)
        
        # Should call base strategy once
        assert mock_strategy.call_count == 1
        assert reranked.qid == 'q1'
    
    def test_multi_query_calls_strategy_per_subquery(self):
        """Test that multi-query reranking calls base strategy for each sub-query."""
        mock_strategy = MockRerankStrategy()
        aggregator = RRFAggregator()
        
        result = create_test_result('q1', 'original query', [('d1', 1.0), ('d2', 0.5)])
        attach_sub_queries(result, ['sub1', 'sub2', 'sub3'])
        
        reranked = _test_simulate_multi_query_rerank(result, mock_strategy, aggregator)
        
        # Should call base strategy 3 times (once per sub-query)
        assert mock_strategy.call_count == 3
        # All sub-queries should have been processed
        assert 'sub1' in mock_strategy.last_queries
        assert 'sub2' in mock_strategy.last_queries
        assert 'sub3' in mock_strategy.last_queries
    
    def test_multi_query_aggregates_results(self):
        """Test that results from sub-queries are aggregated."""
        mock_strategy = MockRerankStrategy()
        aggregator = RRFAggregator(k=60)
        
        result = create_test_result('q1', 'original query', [
            ('d1', 1.0), ('d2', 0.8), ('d3', 0.6)
        ])
        attach_sub_queries(result, ['sub1', 'sub2'])
        
        reranked = _test_simulate_multi_query_rerank(result, mock_strategy, aggregator)
        
        # Should have aggregated results
        assert reranked.qid == 'q1'
        assert len(reranked.hits) == 3  # All docs should be present
        # All docs should have RRF scores
        for hit in reranked.hits:
            assert 'score' in hit
            assert hit['score'] > 0
    
    def test_original_query_preserved(self):
        """Test that the original query is preserved in aggregated result."""
        mock_strategy = MockRerankStrategy()
        aggregator = RRFAggregator()
        
        result = create_test_result('q1', 'my original query', [('d1', 1.0)])
        attach_sub_queries(result, ['sub1', 'sub2'])
        
        reranked = _test_simulate_multi_query_rerank(result, mock_strategy, aggregator)
        
        assert reranked.query == 'my original query'


def run_tests():
    """Run all tests and report results."""
    test_classes = [TestAttachSubQueries, TestMultiQueryLogic]
    
    total = 0
    passed = 0
    failed = []
    
    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total += 1
                try:
                    getattr(instance, method_name)()
                    passed += 1
                    print(f"✓ {test_class.__name__}.{method_name}")
                except Exception as e:
                    failed.append((test_class.__name__, method_name, str(e)))
                    print(f"✗ {test_class.__name__}.{method_name}: {e}")
    
    print(f"\n{passed}/{total} tests passed")
    if failed:
        print("Failed tests:")
        for cls, method, error in failed:
            print(f"  - {cls}.{method}: {error}")
    
    return len(failed) == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
