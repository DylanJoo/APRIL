"""
Unit tests for the ExampleFormatter and example integration in formatters.

Note: Examples are currently only supported for pointwise, pairwise, and judge paradigms.
Listwise and setwise do not support examples yet.
"""
import sys
import argparse
sys.path.insert(0, 'src')


def test_example_formatter_basic():
    """Test basic example formatting for supported paradigms."""
    from autollmrerank.prompt_builder.formatter.example import ExampleFormatter, Example
    
    examples_data = [
        {'query': 'What is Python?', 'document': 'Python is a programming language.', 'label': 'relevant', 'score': 5},
        {'query': 'What is Python?', 'document': 'Java is a coffee.', 'label': 'irrelevant', 'score': 1}
    ]
    
    formatter = ExampleFormatter(examples=examples_data)
    
    # Pairwise
    result = formatter.format('pairwise')
    assert 'Example comparison' in result, "Pairwise format should show comparison example"
    assert 'Passage [1]' in result, "Pairwise format should have passage identifiers"
    
    # Pointwise
    result = formatter.format('pointwise')
    assert 'Example assessments' in result, "Pointwise format should show assessments"
    assert 'Answer: Yes' in result or 'Answer: No' in result, "Pointwise should have Yes/No answers"
    
    # Judge
    result = formatter.format('judge')
    assert 'Example ratings' in result, "Judge format should show ratings"
    assert 'Rating:' in result, "Judge format should have ratings"
    
    print("✓ Basic example formatting tests passed")


def test_unsupported_paradigms():
    """Test that listwise and setwise paradigms return empty string for examples."""
    from autollmrerank.prompt_builder.formatter.example import ExampleFormatter
    
    examples_data = [
        {'query': 'What is Python?', 'document': 'Python is a programming language.', 'label': 'relevant', 'score': 5},
        {'query': 'What is Python?', 'document': 'Java is a coffee.', 'label': 'irrelevant', 'score': 1}
    ]
    
    formatter = ExampleFormatter(examples=examples_data)
    assert formatter.format('listwise') == '', "Listwise should return empty (not supported)"
    assert formatter.format('setwise') == '', "Setwise should return empty (not supported)"
    
    print("✓ Unsupported paradigms tests passed")


def test_example_formatter_empty():
    """Test ExampleFormatter with no examples."""
    from autollmrerank.prompt_builder.formatter.example import ExampleFormatter
    
    # No examples provided
    formatter = ExampleFormatter(examples=None)
    assert formatter.format('pointwise') == '', "Should return empty with no examples"
    
    # Empty list
    formatter = ExampleFormatter(examples=[])
    assert formatter.format('pairwise') == '', "Should return empty with empty list"
    
    print("✓ Empty examples tests passed")


def test_formatter_integration_simplified():
    """Test that formatters properly integrate examples with simplified config."""
    from autollmrerank.prompt_builder.formatter.pairwise import PairwiseFormatter
    from autollmrerank.prompt_builder.formatter.pointwise import PointwiseFormatter
    from autollmrerank.prompt_builder.formatter.judge import JudgeFormatter
    from autollmrerank.prompt_builder.formatter.listwise import ListwiseFormatter
    from autollmrerank.prompt_builder.formatter.setwise import SetwiseFormatter
    
    examples_data = [
        {'query': 'Test query', 'document': 'Test document', 'label': 'relevant', 'score': 5},
        {'query': 'Test query', 'document': 'Irrelevant doc', 'label': 'irrelevant', 'score': 1}
    ]
    
    # New simplified config: examples is just a list
    config_with_examples = argparse.Namespace(
        use_alphabetical=False,
        variable_passages=True,
        max_doc_length=50,
        examples=examples_data  # Direct list, not nested dict
    )
    
    config_without_examples = argparse.Namespace(
        use_alphabetical=False,
        variable_passages=True,
        max_doc_length=50,
        examples=None
    )
    
    # Test ListwiseFormatter - should NOT include examples (not supported)
    formatter = ListwiseFormatter(config_with_examples)
    prefix = formatter.prefix(query='Q', doc_list=[{}, {}])
    assert 'Example' not in prefix, "Listwise should NOT include examples"
    
    # Test SetwiseFormatter - should NOT include examples (not supported)
    formatter = SetwiseFormatter(config_with_examples)
    prefix = formatter.prefix(query='Q', idx_pairs=[[0,1]])
    assert 'Example' not in prefix, "Setwise should NOT include examples"
    
    # Test PairwiseFormatter - should include examples
    formatter = PairwiseFormatter(config_with_examples)
    prefix = formatter.prefix(query='Q')
    assert 'Example comparison' in prefix, "Pairwise should include examples"
    
    # Test PointwiseFormatter - should include examples
    formatter = PointwiseFormatter(config_with_examples)
    prefix = formatter.prefix()
    assert 'Example assessments' in prefix, "Pointwise should include examples"
    
    # Test JudgeFormatter - should include examples
    formatter = JudgeFormatter(config_with_examples)
    prefix = formatter.prefix()
    assert 'Example ratings' in prefix, "Judge should include examples"
    
    # Test without examples config
    formatter = PairwiseFormatter(config_without_examples)
    prefix = formatter.prefix(query='Q')
    assert 'Example' not in prefix, "Without config should not have examples"
    
    print("✓ Formatter integration tests passed")


def test_formatter_paradigm_attribute():
    """Test that each formatter has the correct paradigm attribute."""
    from autollmrerank.prompt_builder.formatter.listwise import ListwiseFormatter
    from autollmrerank.prompt_builder.formatter.pairwise import PairwiseFormatter
    from autollmrerank.prompt_builder.formatter.pointwise import PointwiseFormatter
    from autollmrerank.prompt_builder.formatter.setwise import SetwiseFormatter
    from autollmrerank.prompt_builder.formatter.judge import JudgeFormatter
    
    config = argparse.Namespace(
        use_alphabetical=False,
        variable_passages=True,
        max_doc_length=50,
        examples=None
    )
    
    assert ListwiseFormatter(config).paradigm == 'listwise'
    assert PairwiseFormatter(config).paradigm == 'pairwise'
    assert PointwiseFormatter(config).paradigm == 'pointwise'
    assert SetwiseFormatter(config).paradigm == 'setwise'
    assert JudgeFormatter(config).paradigm == 'judge'
    
    print("✓ All paradigm attribute tests passed")


if __name__ == '__main__':
    test_example_formatter_basic()
    test_unsupported_paradigms()
    test_example_formatter_empty()
    test_formatter_integration_simplified()
    test_formatter_paradigm_attribute()
    print("\n✓✓✓ All tests passed! ✓✓✓")
