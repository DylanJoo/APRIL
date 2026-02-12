"""
Example formatter for including query-document pairs in prompts.

This module provides formatting for examples that demonstrate relevant and irrelevant 
query-document pairs. These examples can be included in prompts for ranking paradigms 
to guide the LLM's behavior (few-shot learning).

Currently supported paradigms for examples:
    - pointwise: In-context examples showing Yes/No relevance assessments
    - pairwise: Examples using one positive and one negative document to demonstrate comparison
    - judge: Examples showing rating scale assessments

Note: Listwise and setwise paradigms do not currently support examples
as they involve complex multi-document ranking that is harder to demonstrate with examples.

Configuration:
    Simply provide a list of example dictionaries in the config:
    
    examples:
      - query: "What is the capital of France?"
        document: "Paris is the capital city of France."
        label: "relevant"
      - query: "What is the capital of France?"
        document: "The Eiffel Tower is tall."
        label: "irrelevant"
"""
from typing import List, Optional, Dict, Union


class Example:
    """Represents a single query-document example with relevance label."""
    
    def __init__(self, query: str, document: str, label: str, score: Optional[float] = None):
        """
        Args:
            query: The example query text
            document: The example document text
            label: Relevance label (e.g., 'relevant', 'irrelevant', 'highly_relevant')
            score: Optional relevance score (e.g., 0-5 scale)
        """
        self.query = query
        self.document = document
        self.label = label
        self.score = score


class ExampleFormatter:
    """
    Formats examples for inclusion in prompts.
    
    This class formats and includes examples in ranking prompts using paradigm-specific
    formatting automatically based on the ranking paradigm type.
    """
    
    def __init__(
        self, 
        examples: Optional[List[Dict]] = None,
        max_doc_length: Optional[int] = None
    ):
        """
        Args:
            examples: List of example dictionaries with keys: query, document, label, score (optional)
            max_doc_length: Maximum length of document text in examples (words)
        """
        self.max_doc_length = max_doc_length
        
        # Convert dict examples to Example objects
        self._examples = []
        if examples:
            for ex in examples:
                self._examples.append(Example(
                    query=ex.get('query', ''),
                    document=self._truncate_doc(ex.get('document', '')),
                    label=ex.get('label', 'relevant'),
                    score=ex.get('score')
                ))
    
    def _truncate_doc(self, doc: str) -> str:
        """Truncate document to max_doc_length words."""
        if self.max_doc_length is not None:
            return " ".join(doc.split()[:self.max_doc_length])
        return doc
    
    @property
    def has_examples(self) -> bool:
        """Check if examples are available."""
        return len(self._examples) > 0
    
    def format_for_pairwise(self) -> str:
        """
        Format examples specifically for pairwise comparison tasks.
        
        Returns examples showing which document is more relevant in a pair.
        """
        if not self.has_examples:
            return ""
        
        relevant = [ex for ex in self._examples if ex.label in ['relevant', 'highly_relevant']]
        irrelevant = [ex for ex in self._examples if ex.label == 'irrelevant']
        
        if relevant and irrelevant:
            rel_ex = relevant[0]
            irrel_ex = irrelevant[0]
            
            return (
                "Example comparison:\n"
                f"Query: {rel_ex.query}\n"
                f"Passage [1]: {rel_ex.document}\n"
                f"Passage [2]: {irrel_ex.document}\n"
                f"Answer: Yes (Passage [1] is more relevant)\n\n"
            )
        return ""
    
    def format_for_pointwise(self) -> str:
        """
        Format examples specifically for pointwise relevance assessment.
        
        Returns examples showing relevance labels for individual documents.
        """
        if not self.has_examples:
            return ""
        
        lines = ["Example assessments:\n"]
        
        for ex in self._examples:
            label = "Yes" if ex.label in ['relevant', 'highly_relevant'] else "No"
            lines.append(f"Query: {ex.query}")
            lines.append(f"Passage: {ex.document}")
            lines.append(f"Answer: {label}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _get_default_score(self, label: str) -> int:
        """Get default score based on relevance label."""
        if label == 'highly_relevant':
            return 5
        elif label == 'relevant':
            return 3
        else:
            return 1

    def format_for_judge(self) -> str:
        """
        Format examples specifically for judge/rating tasks.
        
        Returns examples with scores demonstrating the rating scale.
        """
        if not self.has_examples:
            return ""
        
        lines = ["Example ratings:\n"]
        
        for ex in self._examples:
            score = ex.score if ex.score is not None else self._get_default_score(ex.label)
            lines.append(f"Question: {ex.query}")
            lines.append(f"Context: {ex.document}")
            lines.append(f"Rating: {score}")
            lines.append("")
        
        return "\n".join(lines)
    
    def format(self, paradigm: str = 'pointwise') -> str:
        """
        Format examples based on the paradigm type.
        
        Args:
            paradigm: The ranking paradigm. Currently supported: 'pointwise', 'pairwise', 'judge'.
                     'listwise' and 'setwise' are not supported and will return empty string.
        
        Returns:
            Formatted example string ready for inclusion in prompt
        """
        if not self.has_examples:
            return ""
        
        # Listwise and setwise do not support examples
        if paradigm in ['listwise', 'setwise']:
            return ""
        
        # Use paradigm-specific formatting
        paradigm_formatters = {
            'pairwise': self.format_for_pairwise,
            'pointwise': self.format_for_pointwise,
            'judge': self.format_for_judge,
        }
        formatter = paradigm_formatters.get(paradigm)
        if formatter:
            return formatter()
        
        return ""
