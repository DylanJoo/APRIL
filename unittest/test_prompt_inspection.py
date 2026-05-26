"""
Inspect the final prompt fed to the LLM for every re-ranking method.
No LLM generation is performed — only prompt construction is tested.

Run with:
    pytest test_prompt_inspection.py -s -v
"""

import pytest
from unittest.mock import MagicMock, patch
from autollmrerank.config_manager import ConfigManager
from autollmrerank.prompt_builder import PromptBuilder
from autollmrerank.utils import Result


# ── Mock tokenizer (no model download) ────────────────────────────────────────

class MockTokenizer:
    """Simulates a chat-capable tokenizer without loading any model weights."""
    chat_template = "system"  # makes system_message_supported = True in PromptBuilder

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        parts = []
        for msg in messages:
            content = msg.get("content") or ""
            parts.append(f"<|{msg['role']}|>\n{content}")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "\n".join(parts)

    def __call__(self, text, return_tensors=None):
        out = MagicMock()
        out.input_ids.shape = [1, len(text.split())]
        return out


# ── Dummy query and documents ──────────────────────────────────────────────────

QUERY = "What causes climate change?"

DOCS = [
    {"title": "Greenhouse Gases",    "contents": "Carbon dioxide and methane trap heat in the atmosphere."},
    {"title": "Fossil Fuels",        "contents": "Burning coal and oil releases large amounts of CO2."},
    {"title": "Deforestation",       "contents": "Cutting forests reduces Earth's ability to absorb carbon."},
    {"title": "Industrial Emission", "contents": "Industry emits greenhouse gases as byproducts of energy use."},
    {"title": "Ocean Warming",       "contents": "Warmer oceans release stored CO2 and intensify weather patterns."},
]


def _make_result() -> Result:
    hits = [
        {"docid": f"doc{i}", "score": float(5 - i), "content_dict": doc}
        for i, doc in enumerate(DOCS)
    ]
    return Result(qid="q1", query=QUERY, hits=hits)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_builder(config) -> PromptBuilder:
    with patch(
        "autollmrerank.prompt_builder.base.AutoTokenizer.from_pretrained",
        return_value=MockTokenizer(),
    ):
        return PromptBuilder(config=config)


def _print_prompt(name: str, prompt):
    sep = "=" * 72
    first = prompt[0] if isinstance(prompt, list) else prompt
    note = f"  [returns {len(prompt)} prompts — showing prompt[0]]" if isinstance(prompt, list) else ""
    print(f"\n{sep}")
    print(f"  METHOD: {name}{note}")
    print(sep)
    print(first)
    print(sep)


def _first(prompt) -> str:
    return prompt[0] if isinstance(prompt, list) else prompt


# ── Listwise ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode,use_alpha", [
    ("RankGPT",    False),
    ("RankZephyr", False),
    ("RankFirst",  True),
])
def test_listwise(mode, use_alpha):
    config = ConfigManager(
        rerank_mode=mode,
        use_alphabetical=use_alpha,
        variable_passages=True,
    ).get_config()
    pb = _make_builder(config)

    prompt = pb.create_prompt(_make_result(), rank_start=0, rank_end=5)
    _print_prompt(mode, prompt)

    assert QUERY in _first(prompt)


# ── Pairwise ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode,idx_pairs", [
    # PairTopK (bubble): single comparison per step
    ("PairTopK",        [(4, 3)]),
    # PairMaxHeapTopK (heap): parent vs two children, both directions
    ("PairMaxHeapTopK", [(0, 1), (1, 0)]),
    # PairAll: subset of all pairs (full set is n*(n-1))
    ("PairAll",         [(0, 1), (1, 0), (0, 2), (2, 0)]),
])
def test_pairwise(mode, idx_pairs):
    config = ConfigManager(
        rerank_mode=mode,
        use_alphabetical=False,
        variable_passages=False,
        window_size=2,
        step_size=1,
    ).get_config()
    pb = _make_builder(config)

    prompt = pb.create_prompt(_make_result(), rank_start=0, rank_end=5, idx_pairs=idx_pairs)
    _print_prompt(mode, prompt)

    assert QUERY in _first(prompt)


# ── Setwise ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode,use_alpha", [
    ("SetTopK",        False),
    ("SetMaxHeapTopK", True),
])
def test_setwise(mode, use_alpha):
    config = ConfigManager(
        rerank_mode=mode,
        use_alphabetical=use_alpha,
        variable_passages=False,
        window_size=5,
        step_size=1,
    ).get_config()
    pb = _make_builder(config)

    # SetBubbleTopK/SetMaxHeapTopK pass idx_pairs as a list containing one tuple of window indices
    prompt = pb.create_prompt(
        _make_result(), rank_start=0, rank_end=5, idx_pairs=[(0, 1, 2, 3, 4)]
    )
    _print_prompt(mode, prompt)

    assert QUERY in _first(prompt)


# ── Pointwise / Judge / Lancer ─────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["Point", "Judge", "Lancer"])
def test_pointlike(mode):
    config = ConfigManager(
        rerank_mode=mode,
        use_alphabetical=False,
        variable_passages=False,
    ).get_config()
    pb = _make_builder(config)

    # These formatters return one prompt per document (list of N prompts)
    prompt = pb.create_prompt(_make_result(), rank_start=0, rank_end=5)
    _print_prompt(mode, prompt)

    assert QUERY in _first(prompt)
