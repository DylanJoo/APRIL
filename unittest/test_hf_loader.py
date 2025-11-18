"""
Test script for HuggingFace dataset loader.
This script tests the load_hf function with mock data.
"""
import sys
import os

# Mock the datasets library if the actual datasets aren't available
class MockDataset:
    def __init__(self, data):
        self.data = data
        
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def mock_load_dataset(dataset_name, config, split):
    """Mock HuggingFace load_dataset for testing."""
    if 'nano-beir' in dataset_name and 'corpus' not in dataset_name:
        # Mock queries dataset
        return MockDataset([
            {'query_id': '1', 'query_texts': 'What is information retrieval?'},
            {'query_id': '2', 'query_texts': 'How does neural ranking work?'},
            {'query_id': '3', 'query_texts': 'What is BERT?'},
        ])
    elif 'corpus' in dataset_name:
        # Mock corpus dataset
        return MockDataset([
            {'docid': 'doc1', 'title': 'Information Retrieval', 'text': 'Information retrieval is the process of obtaining information system resources.'},
            {'docid': 'doc2', 'title': 'Neural Ranking', 'text': 'Neural ranking models use deep learning for document ranking.'},
            {'docid': 'doc3', 'title': 'BERT Model', 'text': 'BERT is a transformer-based machine learning technique.'},
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# Patch the datasets module
class MockDatasetsModule:
    @staticmethod
    def load_dataset(dataset_name, config, split):
        return mock_load_dataset(dataset_name, config, split)

sys.modules['datasets'] = MockDatasetsModule()

# Now import and test the loader
from reranking import loader

def test_load_hf():
    """Test the load_hf function with mock data."""
    print("Testing load_hf function...")
    
    # Test loading both queries and corpus
    corpus, queries, qrels = loader.load_hf(
        queries_dataset_name='DylanJHJ/nano-beir',
        corpus_dataset_name='DylanJHJ/nano-beir-corpus',
        subset='nq',
        query_split='test',
        corpus_split='train',
    )
    
    print(f"\nLoaded {len(queries)} queries:")
    for qid, query_text in queries.items():
        print(f"  {qid}: {query_text[:50]}...")
    
    print(f"\nLoaded {len(corpus)} documents:")
    for docid, doc_dict in list(corpus.items())[:3]:
        print(f"  {docid}: {doc_dict['contents'][:50]}...")
    
    # Verify the structure
    assert isinstance(corpus, dict), "Corpus should be a dict"
    assert isinstance(queries, dict), "Queries should be a dict"
    assert isinstance(qrels, dict), "Qrels should be a dict"
    
    # Verify queries structure
    assert len(queries) == 3, "Should have 3 queries"
    assert all(isinstance(qid, str) for qid in queries.keys()), "Query IDs should be strings"
    assert all(isinstance(text, str) for text in queries.values()), "Query texts should be strings"
    
    # Verify corpus structure
    assert len(corpus) == 3, "Should have 3 documents"
    assert all(isinstance(docid, str) for docid in corpus.keys()), "Doc IDs should be strings"
    assert all(isinstance(doc, dict) and 'contents' in doc for doc in corpus.values()), "Each doc should have 'contents'"
    
    # Test ignore_corpus flag
    corpus_none, queries2, qrels2 = loader.load_hf(
        queries_dataset_name='DylanJHJ/nano-beir',
        corpus_dataset_name='DylanJHJ/nano-beir-corpus',
        subset='nq',
        ignore_corpus=True,
    )
    
    assert corpus_none is None, "Corpus should be None when ignore_corpus=True"
    assert len(queries2) == 3, "Should still have queries"
    
    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_load_hf()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
