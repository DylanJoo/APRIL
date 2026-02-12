"""
Unit tests for the ExampleFormatter and example integration in formatters.
"""
import sys
import argparse
sys.path.insert(0, 'src')


def test_example_formatter_strategies():
    """Test different example formatting strategies."""
    from autollmrerank.prompt_builder.formatter.example import ExampleFormatter, Example
    
    examples_data = [
        {'query': 'What is Python?', 'document': 'Python is a programming language.', 'label': 'relevant', 'score': 5},
        {'query': 'What is Python?', 'document': 'Java is a coffee.', 'label': 'irrelevant', 'score': 1}
    ]
    
    # Test 'none' strategy returns empty string
    formatter = ExampleFormatter(examples=examples_data, strategy='none')
    assert formatter.format('listwise') == '', "None strategy should return empty string"
    
    # Test 'block' strategy returns formatted block
    formatter = ExampleFormatter(examples=examples_data, strategy='block')
    result = formatter.format('listwise')
    assert 'Examples of relevance assessment' in result, "Block format should have header"
    assert 'Example 1:' in result, "Block format should have numbered examples"
    assert 'Relevant' in result, "Block format should show relevance assessment"
    
    # Test 'inline' strategy returns inline description
    formatter = ExampleFormatter(examples=examples_data, strategy='inline')
    result = formatter.format('listwise')
    assert 'For example' in result, "Inline format should start with example phrase"
    assert 'relevant passage' in result, "Inline format should mention relevant passage"
    
    # Test 'interleaved' strategy for different paradigms
    formatter = ExampleFormatter(examples=examples_data, strategy='interleaved')
    
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
    
    print("✓ All strategy tests passed")


def test_example_formatter_empty():
    """Test ExampleFormatter with no examples."""
    from autollmrerank.prompt_builder.formatter.example import ExampleFormatter
    
    # No examples provided
    formatter = ExampleFormatter(examples=None, strategy='block')
    assert formatter.format('listwise') == '', "Should return empty with no examples"
    
    # Empty list
    formatter = ExampleFormatter(examples=[], strategy='block')
    assert formatter.format('listwise') == '', "Should return empty with empty list"
    
    print("✓ Empty examples tests passed")


def test_example_formatter_max_examples():
    """Test max_examples parameter."""
    from autollmrerank.prompt_builder.formatter.example import ExampleFormatter
    
    examples_data = [
        {'query': 'Q1', 'document': 'D1', 'label': 'relevant'},
        {'query': 'Q2', 'document': 'D2', 'label': 'relevant'},
        {'query': 'Q3', 'document': 'D3', 'label': 'relevant'},
    ]
    
    # Only 1 example should be used
    formatter = ExampleFormatter(examples=examples_data, strategy='block', max_examples=1)
    assert formatter._examples is not None
    assert len(formatter._examples) == 1, "Should only have 1 example"
    
    # All 3 examples
    formatter = ExampleFormatter(examples=examples_data, strategy='block', max_examples=5)
    assert len(formatter._examples) == 3, "Should have all 3 examples"
    
    print("✓ Max examples tests passed")


def test_formatter_integration():
    """Test that formatters properly integrate examples."""
    from autollmrerank.prompt_builder.formatter.listwise import ListwiseFormatter
    from autollmrerank.prompt_builder.formatter.pairwise import PairwiseFormatter
    from autollmrerank.prompt_builder.formatter.pointwise import PointwiseFormatter
    from autollmrerank.prompt_builder.formatter.setwise import SetwiseFormatter
    from autollmrerank.prompt_builder.formatter.judge import JudgeFormatter
    
    examples_data = [
        {'query': 'Test query', 'document': 'Test document', 'label': 'relevant', 'score': 5}
    ]
    
    config_with_examples = argparse.Namespace(
        use_alphabetical=False,
        variable_passages=True,
        max_doc_length=50,
        examples=argparse.Namespace(
            strategy='block',
            max_examples=2,
            data=examples_data
        )
    )
    
    config_without_examples = argparse.Namespace(
        use_alphabetical=False,
        variable_passages=True,
        max_doc_length=50,
        examples=None
    )
    
    # Test ListwiseFormatter
    formatter = ListwiseFormatter(config_with_examples)
    prefix = formatter.prefix(query='Q', doc_list=[{}, {}])
    assert 'Examples of relevance assessment' in prefix, "Listwise should include examples"
    
    formatter = ListwiseFormatter(config_without_examples)
    prefix = formatter.prefix(query='Q', doc_list=[{}, {}])
    assert 'Examples of relevance assessment' not in prefix, "Listwise without config should not have examples"
    
    # Test PairwiseFormatter
    formatter = PairwiseFormatter(config_with_examples)
    prefix = formatter.prefix(query='Q')
    assert 'Examples of relevance assessment' in prefix, "Pairwise should include examples"
    
    # Test PointwiseFormatter
    formatter = PointwiseFormatter(config_with_examples)
    prefix = formatter.prefix()
    assert 'Examples of relevance assessment' in prefix, "Pointwise should include examples"
    
    # Test SetwiseFormatter
    formatter = SetwiseFormatter(config_with_examples)
    prefix = formatter.prefix(query='Q', idx_pairs=[[0,1]])
    assert 'Examples of relevance assessment' in prefix, "Setwise should include examples"
    
    # Test JudgeFormatter
    formatter = JudgeFormatter(config_with_examples)
    prefix = formatter.prefix()
    assert 'Examples of relevance assessment' in prefix, "Judge should include examples"
    
    print("✓ All formatter integration tests passed")


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


def test_invalid_strategy():
    """Test that invalid strategy raises an error."""
    from autollmrerank.prompt_builder.formatter.example import ExampleFormatter
    
    try:
        ExampleFormatter(strategy='invalid_strategy')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'Unsupported example strategy' in str(e)
    
    print("✓ Invalid strategy test passed")


if __name__ == '__main__':
    test_example_formatter_strategies()
    test_example_formatter_empty()
    test_example_formatter_max_examples()
    test_formatter_integration()
    test_formatter_paradigm_attribute()
    test_invalid_strategy()
    print("\n✓✓✓ All tests passed! ✓✓✓")
