"""
Example formatter for including query-document pairs in prompts.

This module provides different strategies for formatting examples that demonstrate
relevant and irrelevant query-document pairs. These examples can be included in
the prompts for ranking paradigms to guide the LLM's behavior.

Currently supported paradigms for examples:
    - pointwise: In-context examples showing Yes/No relevance assessments
    - pairwise: Examples using one positive and one negative document to demonstrate comparison
    - judge: Examples showing rating scale assessments

Note: Listwise and setwise paradigms do not currently support examples
as they involve complex multi-document ranking that is harder to demonstrate with examples.

Supported strategies:
    - inline: Include examples directly in the instruction text
    - block: Include examples as a separate block before the main content
    - interleaved: Paradigm-specific formatting (recommended for pointwise, pairwise, judge)
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
    
    This class provides various strategies for formatting and including examples
    in ranking prompts. The strategy determines how examples are presented to
    guide the LLM's understanding of relevance.
    """
    
    SUPPORTED_STRATEGIES = ['inline', 'block', 'interleaved', 'none']
    
    # Preview length for truncating document text in inline strategy
    INLINE_DOC_PREVIEW_LENGTH = 200
    
    def __init__(
        self, 
        examples: Optional[List[Dict]] = None,
        strategy: str = 'none',
        max_examples: int = 2,
        max_doc_length: Optional[int] = None
    ):
        """
        Args:
            examples: List of example dictionaries with keys: query, document, label, score (optional)
            strategy: How to format examples ('inline', 'block', 'interleaved', 'none')
            max_examples: Maximum number of examples to include
            max_doc_length: Maximum length of document text in examples (words)
        """
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported example strategy: {strategy}. "
                f"Supported strategies: {self.SUPPORTED_STRATEGIES}"
            )
        
        self.strategy = strategy
        self.max_examples = max_examples
        self.max_doc_length = max_doc_length
        
        # Convert dict examples to Example objects
        self._examples = []
        if examples:
            for ex in examples[:max_examples]:
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
        """Check if examples are available and strategy is not 'none'."""
        return len(self._examples) > 0 and self.strategy != 'none'
    
    def format_inline(self, paradigm: str = 'listwise') -> str:
        """
        Format examples for inline inclusion in instruction text.
        
        Returns a brief mention of what constitutes relevant/irrelevant documents.
        """
        if not self.has_examples:
            return ""
        
        relevant_examples = [ex for ex in self._examples if ex.label in ['relevant', 'highly_relevant']]
        irrelevant_examples = [ex for ex in self._examples if ex.label == 'irrelevant']
        
        parts = []
        if relevant_examples:
            ex = relevant_examples[0]
            parts.append(
                f"For example, for the query \"{ex.query}\", "
                f"a relevant passage would be: \"{ex.document[:self.INLINE_DOC_PREVIEW_LENGTH]}...\""
            )
        if irrelevant_examples:
            ex = irrelevant_examples[0]
            parts.append(
                f"An irrelevant passage would be: \"{ex.document[:self.INLINE_DOC_PREVIEW_LENGTH]}...\""
            )
        
        return " ".join(parts)
    
    def format_block(self, paradigm: str = 'listwise') -> str:
        """
        Format examples as a separate block to be included in the prompt.
        
        Returns a formatted block containing all examples with clear labeling.
        """
        if not self.has_examples:
            return ""
        
        lines = ["Examples of relevance assessment:\n"]
        
        for i, ex in enumerate(self._examples, start=1):
            label_text = "Relevant" if ex.label in ['relevant', 'highly_relevant'] else "Irrelevant"
            score_text = f" (Score: {ex.score})" if ex.score is not None else ""
            
            lines.append(f"Example {i}:")
            lines.append(f"  Query: {ex.query}")
            lines.append(f"  Document: {ex.document}")
            lines.append(f"  Assessment: {label_text}{score_text}")
            lines.append("")
        
        return "\n".join(lines)
    
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
    
    def format_for_listwise(self) -> str:
        """
        Format examples specifically for listwise ranking tasks.
        
        Note: Listwise examples are not currently supported as they involve
        complex multi-document ranking. Returns empty string.
        """
        # Listwise examples are not supported yet
        return ""
    
    def format_for_setwise(self) -> str:
        """
        Format examples specifically for setwise comparison tasks.
        
        Note: Setwise examples are not currently supported as they involve
        selecting from multiple documents. Returns empty string.
        """
        # Setwise examples are not supported yet
        return ""
    
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
        Format examples based on the current strategy and paradigm.
        
        Args:
            paradigm: The ranking paradigm. Currently supported: 'pointwise', 'pairwise', 'judge'.
                     'listwise' and 'setwise' are not supported and will return empty string.
                     Note: The default value changed from 'listwise' to 'pointwise' since 
                     listwise is not supported. This should not affect normal usage as the
                     paradigm is typically passed explicitly via the formatter's paradigm attribute.
        
        Returns:
            Formatted example string ready for inclusion in prompt
        """
        if self.strategy == 'none' or not self.has_examples:
            return ""
        
        # Listwise and setwise do not support examples
        if paradigm in ['listwise', 'setwise']:
            return ""
        
        if self.strategy == 'inline':
            return self.format_inline(paradigm)
        elif self.strategy == 'block':
            return self.format_block(paradigm)
        elif self.strategy == 'interleaved':
            # For interleaved, use paradigm-specific formatting
            paradigm_formatters = {
                'pairwise': self.format_for_pairwise,
                'pointwise': self.format_for_pointwise,
                'judge': self.format_for_judge,
            }
            formatter = paradigm_formatters.get(paradigm, self.format_block)
            return formatter()
        
        return ""
