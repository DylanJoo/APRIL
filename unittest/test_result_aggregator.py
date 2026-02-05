"""
Unit tests for the result aggregation module.
"""
import sys
import os

# Add src directory to path dynamically based on script location
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), 'src')
sys.path.insert(0, src_dir)

from autollmrerank.result_aggregator import (
    ResultAggregator,
    PassThroughAggregator,
    RRFAggregator,
    CoverageAggregator,
    MMRAggregator,
)
from autollmrerank.utils import Result


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


class TestPassThroughAggregator:
    """Tests for PassThroughAggregator."""
    
    def test_returns_first_result(self):
        """Test that passthrough returns first result unchanged."""
        result1 = create_test_result('q1', 'query1', [('d1', 1.0), ('d2', 0.5)])
        result2 = create_test_result('q1', 'query1', [('d3', 1.0), ('d4', 0.5)])
        
        aggregator = PassThroughAggregator()
        aggregated = aggregator.aggregate([result1, result2])
        
        assert len(aggregated.hits) == 2
        assert aggregated.hits[0]['docid'] == 'd1'
        assert aggregated.hits[1]['docid'] == 'd2'
    
    def test_updates_query_if_provided(self):
        """Test that original_query is used if provided."""
        result = create_test_result('q1', 'sub_query', [('d1', 1.0)])
        
        aggregator = PassThroughAggregator()
        aggregated = aggregator.aggregate(
            [result], 
            original_query='original query',
            qid='original_qid'
        )
        
        assert aggregated.query == 'original query'
        assert aggregated.qid == 'original_qid'


class TestRRFAggregator:
    """Tests for RRFAggregator (Reciprocal Rank Fusion)."""
    
    def test_combines_rankings(self):
        """Test that RRF combines multiple rankings."""
        # Result 1: d1 ranked 1st, d2 ranked 2nd
        result1 = create_test_result('q1', 'sub1', [('d1', 1.0), ('d2', 0.5)])
        # Result 2: d2 ranked 1st, d1 ranked 2nd
        result2 = create_test_result('q1', 'sub2', [('d2', 1.0), ('d1', 0.5)])
        
        aggregator = RRFAggregator(k=60)
        aggregated = aggregator.aggregate([result1, result2])
        
        # Both docs should appear with combined RRF scores
        assert len(aggregated.hits) == 2
        docids = {hit['docid'] for hit in aggregated.hits}
        assert 'd1' in docids
        assert 'd2' in docids
    
    def test_respects_weights(self):
        """Test that weights affect ranking."""
        result1 = create_test_result('q1', 'sub1', [('d1', 1.0)])
        result2 = create_test_result('q1', 'sub2', [('d2', 1.0)])
        
        # Heavily weight first result
        aggregator = RRFAggregator(k=60)
        aggregated = aggregator.aggregate(
            [result1, result2],
            weights=[0.9, 0.1]
        )
        
        # d1 should be ranked higher due to higher weight
        assert aggregated.hits[0]['docid'] == 'd1'
    
    def test_handles_overlapping_docs(self):
        """Test handling of documents appearing in multiple results."""
        result1 = create_test_result('q1', 'sub1', [('d1', 1.0), ('d2', 0.5)])
        result2 = create_test_result('q1', 'sub2', [('d1', 1.0), ('d3', 0.5)])
        
        aggregator = RRFAggregator()
        aggregated = aggregator.aggregate([result1, result2])
        
        # d1 appears in both and should have highest score
        assert aggregated.hits[0]['docid'] == 'd1'


class TestCoverageAggregator:
    """Tests for CoverageAggregator."""
    
    def test_covers_all_aspects(self):
        """Test that coverage aggregator considers all aspects."""
        # Aspect 1: d1 is best
        result1 = create_test_result('q1', 'aspect1', [('d1', 1.0), ('d2', 0.2)])
        # Aspect 2: d2 is best
        result2 = create_test_result('q1', 'aspect2', [('d2', 1.0), ('d1', 0.2)])
        
        aggregator = CoverageAggregator(coverage_weight=0.5)
        aggregated = aggregator.aggregate([result1, result2])
        
        # Both docs should be well-ranked to cover both aspects
        assert len(aggregated.hits) == 2
        docids = [hit['docid'] for hit in aggregated.hits]
        assert 'd1' in docids
        assert 'd2' in docids
    
    def test_top_k_limit(self):
        """Test that top_k limits output."""
        result1 = create_test_result('q1', 'sub1', [
            ('d1', 1.0), ('d2', 0.9), ('d3', 0.8)
        ])
        
        aggregator = CoverageAggregator(top_k=2)
        aggregated = aggregator.aggregate([result1])
        
        assert len(aggregated.hits) == 2


class TestMMRAggregator:
    """Tests for MMRAggregator (Maximal Marginal Relevance)."""
    
    def test_mmr_basic(self):
        """Test basic MMR functionality."""
        result1 = create_test_result('q1', 'sub1', [('d1', 1.0), ('d2', 0.5)])
        result2 = create_test_result('q1', 'sub2', [('d2', 1.0), ('d1', 0.5)])
        
        aggregator = MMRAggregator(lambda_param=0.7)
        aggregated = aggregator.aggregate([result1, result2])
        
        assert len(aggregated.hits) == 2


def run_tests():
    """Run all tests and report results."""
    test_classes = [
        TestPassThroughAggregator,
        TestRRFAggregator,
        TestCoverageAggregator,
        TestMMRAggregator,
    ]
    
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
