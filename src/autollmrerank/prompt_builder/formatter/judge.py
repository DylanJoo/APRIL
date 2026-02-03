"""
JudgeFormatter: Flexible formatter for LLM-as-a-Judge scoring methods.
Supports multiple scoring modes:
- binary: Yes/No relevance (legacy pointwise)
- rating: Rubric-based rating (0-5 scale)
- rubric_binary: Rubric with binary output for specific rating threshold
"""
from typing import List, Optional, Union, Dict, Tuple
from .base import BaseFormatter


# Default rating rubric for relevance judgment (0-5 scale)
DEFAULT_RATING_RUBRIC = """Rating scale:
0 - Not relevant: The passage has nothing to do with the query.
1 - Marginally relevant: The passage mentions related topics but doesn't address the query.
2 - Partially relevant: The passage contains some useful information but is incomplete.
3 - Fairly relevant: The passage addresses the query but may lack depth or specificity.
4 - Highly relevant: The passage provides substantial and useful information for the query.
5 - Perfectly relevant: The passage completely and precisely answers the query."""


class JudgeFormatter(BaseFormatter):
    """
    Formatter for LLM-as-a-Judge scoring with configurable scoring modes.
    
    Scoring modes:
    - 'binary': Simple Yes/No relevance judgment
    - 'rating': Rate on a scale (default 0-5)
    - 'rubric_binary': Use rubric context but output binary (Yes/No for threshold)
    """

    def __init__(self, config):
        super().__init__(config)
        # Judge-specific settings
        self.scoring_mode = getattr(config, 'scoring_mode', 'binary')
        self.rating_scale = getattr(config, 'rating_scale', 5)  # max rating value
        self.rating_threshold = getattr(config, 'rating_threshold', 3)  # for rubric_binary mode
        self.rubric = getattr(config, 'rubric', DEFAULT_RATING_RUBRIC)
        self.include_rubric = getattr(config, 'include_rubric', True)
        
        # Few-shot reference settings
        self.num_references = getattr(config, 'num_references', 0)
        self.reference_structure = getattr(config, 'reference_structure', None)

    def prefix(self, **kwargs) -> str:
        """Generate the prefix based on scoring mode."""
        references = kwargs.get('references', None)
        
        if self.scoring_mode == 'binary':
            prefix = "You are a relevance judge. Your task is to determine if a passage is relevant to the query.\n\n"
        elif self.scoring_mode == 'rating':
            prefix = "You are a relevance judge. Your task is to rate the relevance of a passage to the query.\n\n"
            if self.include_rubric:
                prefix += f"{self.rubric}\n\n"
        elif self.scoring_mode == 'rubric_binary':
            prefix = (
                f"You are a relevance judge. Your task is to determine if a passage meets the relevance threshold.\n"
                f"A passage is considered relevant if it would receive a rating of {self.rating_threshold} or higher.\n\n"
            )
            if self.include_rubric:
                prefix += f"{self.rubric}\n\n"
        else:
            prefix = "You are a relevance judge.\n\n"
        
        # Add few-shot references if provided
        if references and self.num_references > 0:
            prefix += self._format_references(references)
        
        return prefix

    def postfix(self, **kwargs) -> str:
        """Generate the postfix (instruction) based on scoring mode."""
        if self.scoring_mode == 'binary':
            return (
                "Is this passage relevant to the query?\n"
                "Only respond with Yes or No, do not explain.\nAnswer: "
            )
        elif self.scoring_mode == 'rating':
            return (
                f"Rate the relevance of this passage to the query on a scale of 0 to {self.rating_scale}.\n"
                "Only respond with a single number, do not explain.\nRating: "
            )
        elif self.scoring_mode == 'rubric_binary':
            return (
                f"Does this passage meet the relevance threshold (rating >= {self.rating_threshold})?\n"
                "Only respond with Yes or No, do not explain.\nAnswer: "
            )
        else:
            return "Provide your judgment.\nAnswer: "

    def body(self, query: str, doc_list: List[Union[Dict, str]], **kwargs) -> Union[str, List[str]]:
        """Generate prompt body for each document."""
        prompts = []
        doc_list = [self._document_format(doc) for doc in doc_list]
        
        for doc in doc_list:
            prompt = f"Query: {query}\nPassage: {doc}\n\n"
            prompts.append(prompt)
        
        return prompts

    def _format_references(self, references: List[Dict]) -> str:
        """Format few-shot reference examples."""
        if not references:
            return ""
        
        ref_text = "Here are some reference examples:\n\n"
        for i, ref in enumerate(references[:self.num_references], start=1):
            query = ref.get('query', '')
            passage = ref.get('passage', '')
            judgment = ref.get('judgment', '')
            
            if self.scoring_mode == 'rating':
                ref_text += f"Example {i}:\nQuery: {query}\nPassage: {passage}\nRating: {judgment}\n\n"
            else:
                ref_text += f"Example {i}:\nQuery: {query}\nPassage: {passage}\nAnswer: {judgment}\n\n"
        
        ref_text += "Now judge the following:\n\n"
        return ref_text


class JudgeFewShotFormatter(JudgeFormatter):
    """
    Extended Judge formatter with enhanced few-shot reference support.
    
    Reference structures:
    - 'positive_only': Include only positive examples
    - 'with_negative': Include positive and negative examples
    - 'tight': Include examples near the decision boundary
    - 'pseudo': Use model-generated pseudo-references
    """

    def __init__(self, config):
        super().__init__(config)
        self.reference_structure = getattr(config, 'reference_structure', 'positive_only')

    def prefix(self, **kwargs) -> str:
        """Generate prefix with enhanced few-shot reference handling."""
        references = kwargs.get('references', None)
        base_prefix = super().prefix(**kwargs)
        
        if references and self.num_references > 0:
            # Override base reference formatting for specific structures
            structured_refs = self._structure_references(references)
            return self._build_prefix_with_refs(base_prefix, structured_refs)
        
        return base_prefix

    def _structure_references(self, references: List[Dict]) -> List[Dict]:
        """Structure references based on reference_structure setting."""
        if self.reference_structure == 'positive_only':
            return [r for r in references if self._is_positive(r)][:self.num_references]
        elif self.reference_structure == 'with_negative':
            positives = [r for r in references if self._is_positive(r)]
            negatives = [r for r in references if not self._is_positive(r)]
            # Alternate positive and negative examples
            structured = []
            for p, n in zip(positives, negatives):
                structured.extend([p, n])
            return structured[:self.num_references]
        elif self.reference_structure == 'tight':
            # Examples near the decision boundary
            return self._get_tight_references(references)
        else:
            return references[:self.num_references]

    def _is_positive(self, reference: Dict) -> bool:
        """Check if a reference is a positive example."""
        judgment = reference.get('judgment', '')
        if self.scoring_mode == 'rating':
            try:
                return int(judgment) >= self.rating_threshold
            except (ValueError, TypeError):
                return False
        else:
            return str(judgment).lower() in ['yes', 'true', '1']

    def _get_tight_references(self, references: List[Dict]) -> List[Dict]:
        """Get references near the decision boundary for tight sampling."""
        if self.scoring_mode == 'rating':
            # Get examples with ratings near the threshold
            boundary_refs = []
            for r in references:
                try:
                    rating = int(r.get('judgment', 0))
                    if abs(rating - self.rating_threshold) <= 1:
                        boundary_refs.append(r)
                except (ValueError, TypeError):
                    continue
            return boundary_refs[:self.num_references]
        else:
            return references[:self.num_references]

    def _build_prefix_with_refs(self, base_prefix: str, references: List[Dict]) -> str:
        """Build the full prefix with structured references."""
        # Remove any existing reference formatting from base prefix
        if "Here are some reference examples:" in base_prefix:
            base_prefix = base_prefix.split("Here are some reference examples:")[0]
        
        ref_text = self._format_references(references)
        return base_prefix + ref_text
