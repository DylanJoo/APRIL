import copy
from typing import List, Optional, Tuple, Callable, Dict, Union
from ..utils import PromptMode, Result

from ._rank_gpt import TextListParser

class RankParser:
    def __init__(
        self, 
        prompt_mode: PromptMode,
        **kwargs
    ):
        self.prompt_mode = prompt_mode
        self.parser = self._get_parser(prompt_mode, **kwargs)

    def _get_parser(self, prompt_mode: PromptMode, **kwargs) -> Callable:
        parser_map: Dict[PromptMode, Callable] = {
            PromptMode.RANK_GPT: TextListParser,
        }
        if prompt_mode not in parser_map:
            raise ValueError(f"Unsupported prompt mode: {prompt_mode}")
        return parser_map[prompt_mode](**kwargs)

    # [NOTE] parse: response is the input, number as output 
    # [NOTE] update: number to the result object
    # [NOTE] also consider additional info: permutation, out_token_count. Use the origianl for loop
    # for index, (result, (prompt, in_token_count)) in enumerate(zip(results, prompts)):
    def parse_response(
        self, 
        response_texts: List[str], 
        results: List[Result], 
        rank_start: int, 
        rank_end: int, 
    ) -> str:
        assert len(response_texts) == len(results), "Response texts and results must have the same length."

        for index, (response, result) in enumerate(zip(response_texts, results)):
            parsed_result = self.parser.parse_and_update(response, result, rank_start, rank_end)
            results[index] = parsed_result
        return results
