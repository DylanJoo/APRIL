# UMBRELA-style relevance criteria applied to setwise ranking.
# Keeps the exact sentence structure of SetwiseFormatter (same opening line, same
# closing output-format instruction) and weaves in the UMBRELA 0-3 scale and its M/T
# (intent-match / trustworthiness) decomposition as clauses within it, rather than
# stacking separate UMBRELA blocks on top of the untouched setwise text. That
# decomposition -- not just the 0-3 labels -- is what distinguishes Umbrela from the
# plain Judge formatter (see umbrela.py), so it needs to survive here too, just
# expressed as a selection criterion instead of a per-passage score.
from typing import List, Optional, Dict
from .setwise import SetwiseFormatter
from .umbrela import UMBRELA_RELEVANCE_SCALE, UMBRELA_RELEVANCE_INSTRUCTION

class UmbrelaSetwiseFormatter(SetwiseFormatter):

    def prefix(self, query, idx_pairs, **kwargs) -> str:
        n_pairs = len(idx_pairs[0])
        return (
            f"I will provide you with {n_pairs} passages. Read and memorize all carefully. "
            f"Judge each passage against the query: {query}, using the following relevance scale, "
            "an integer from 0 to 3 with the following meanings:\n"
            f"{UMBRELA_RELEVANCE_SCALE}"
            f"{UMBRELA_RELEVANCE_INSTRUCTION}"
        )

    def postfix(self, query: str, doc_list: Optional[List[Dict]] = None, idx_pairs=None, **kwargs) -> str:
        return (
            f"Search Query: {query}\n"
            "For each passage, consider the underlying intent of the search, measure how well "
            "the content matches that intent (M), and measure how trustworthy the passage is (T). "
            "Consider the aspects M and T and the relative importance of each, and decide which passage is the most relevant one to the query. "
            f"Only respond with the {self.id_type} identifier with bracket (e.g., [A]). Do not say any word or explain."
        )
