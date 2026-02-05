"""
Unit tests for the query decomposition module.
"""
import sys
sys.path.insert(0, '/home/runner/work/APRIL/APRIL/src')

from autollmrerank.query_decomposer import (
    QueryDecomposer,
    PassThroughDecomposer,
    DecomposedQuery,
)


class TestDecomposedQuery:
    """Tests for DecomposedQuery dataclass."""
    
    def test_default_weights(self):
        """Test that default weights are equal."""
        query = DecomposedQuery(
            original_query="What is AI?",
            sub_queries=["What is artificial intelligence?", "AI definition"]
        )
        assert query.weights is not None
        assert len(query.weights) == 2
        assert sum(query.weights) == 1.0
        assert all(w == 0.5 for w in query.weights)
    
    def test_custom_weights(self):
        """Test custom weight assignment."""
        query = DecomposedQuery(
            original_query="test query",
            sub_queries=["sub1", "sub2", "sub3"],
            weights=[0.5, 0.3, 0.2]
        )
        assert query.weights == [0.5, 0.3, 0.2]
    
    def test_metadata(self):
        """Test metadata storage."""
        query = DecomposedQuery(
            original_query="test",
            sub_queries=["test"],
            metadata={"source": "test", "version": 1}
        )
        assert query.metadata["source"] == "test"
        assert query.metadata["version"] == 1


class TestPassThroughDecomposer:
    """Tests for PassThroughDecomposer."""
    
    def test_decompose_returns_original(self):
        """Test that passthrough returns original query."""
        decomposer = PassThroughDecomposer()
        result = decomposer.decompose("What is machine learning?")
        
        assert result.original_query == "What is machine learning?"
        assert result.sub_queries == ["What is machine learning?"]
        assert result.weights == [1.0]
        assert result.metadata["method"] == "passthrough"
    
    def test_decompose_batch(self):
        """Test batch decomposition."""
        decomposer = PassThroughDecomposer()
        queries = ["Query 1", "Query 2", "Query 3"]
        results = decomposer.decompose_batch(queries)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.original_query == queries[i]
            assert result.sub_queries == [queries[i]]


def run_tests():
    """Run all tests and report results."""
    test_classes = [TestDecomposedQuery, TestPassThroughDecomposer]
    
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
