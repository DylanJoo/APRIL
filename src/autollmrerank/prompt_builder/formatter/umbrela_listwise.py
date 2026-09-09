# UMBRELA-style relevance criteria applied to RankGPT-style listwise ranking.
# Keeps the exact sentence structure of ListwiseFormatter (same opening line, same
# closing output-format instruction) and weaves in the UMBRELA 0-3 scale and its M/T
# (intent-match / trustworthiness) decomposition as clauses within it, rather than
# stacking separate UMBRELA blocks on top of the untouched listwise text. That
# decomposition -- not just the 0-3 labels -- is what distinguishes Umbrela from the
# plain Judge formatter (see umbrela.py), so it needs to survive here too, just
# expressed as a ranking criterion instead of a per-passage score.
from typing import List, Optional, Dict
from .listwise import ListwiseFormatter
from .umbrela import UMBRELA_RELEVANCE_SCALE, UMBRELA_RELEVANCE_INSTRUCTION

class UmbrelaListwiseFormatter(ListwiseFormatter):

    def prefix(self, query: str, doc_list: Optional[List[Dict]] = None, **kwargs) -> str:
        return (
            f"I will provide you with {len(doc_list)} passages, "
            f"each indicated by a {self.id_type} identifier []. "
            f"Judge each passage against the search query: {query}, using the following relevance scale, "
            "an integer from 0 to 3 with the following meanings:\n"
            f"{UMBRELA_RELEVANCE_SCALE}"
            f"{UMBRELA_RELEVANCE_INSTRUCTION}"
        )

    def postfix(self, query: str, doc_list: Optional[List[Dict]] = None, filtering=False, **kwargs) -> str:
        if filtering:
            return (
                f"Search Query: {query}.\n"
                "For each passage, consider the underlying intent of the search, measure how well "
                "the content matches that intent (M), and measure how trustworthy the passage is (T). "
                f"Consider the aspects M and T and the relative importance of each, and rank the {len(doc_list)} passages above based on their relevance to the search query. "
                f"All the passages should be included and listed using identifiers, "
                f"in descending order of relevance. In addition, add the separation mark to indicate the boundary of relevance, "
                "The output format should be [] > [] > [] > [x] > [] > [], "
                "the first part before [x] is for relevant passages; the second part is for irrelevant ones. "
                f"Only respond with the ranking results, do not say any word or explain."
            )

        return (
            f"Search Query: {query}.\n"
            "For each passage, consider the underlying intent of the search, measure how well "
            "the content matches that intent (M), and measure how trustworthy the passage is (T). "
            f"Consider the aspects M and T and the relative importance of each, and rank the {len(doc_list)} passages above based on their relevance to the search query. "
            f"All the passages should be included and listed using identifiers, "
            f"in descending order of relevance. The output format should be [] > [], "
            f"e.g., {self.example_ordering}, "
            f"Only respond with the ranking results, do not say any word or explain."
        )
