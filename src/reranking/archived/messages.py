ALPH_START_IDX = ord('A')-1
# Listwise re-ranking: list generation
def _add_prefix_prompt(use_alpha, query: str, num: int, variable_passages: bool = False) -> str:
    if use_alpha:
        return f"I will provide you with {num} passages, each indicated by a alphabetical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"
    else:
        return f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"

def _add_post_prompt(use_alpha, query: str, num: int, variable_passages: bool = False) -> str:
    if use_alpha:
        example_ordering = "[B] > [A]" if variable_passages else "[D] > [B]"
    else:
        example_ordering = "[2] > [1]" if variable_passages else "[4] > [2]"
    return f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}, Only respond with the ranking results, do not say any word or explain."

# [material]
# Remember Compare the passages based on their relevance to the search query: {query}.\n"
# "Read and memorize all passages carefully. Your task is to use these passages for multiple comparisons based on their relevance to the search query"
# Pairwise re-ranking: binary token generation
# def _add_prefix_prompt(use_alpha, query: str, num: int, variable_passages: bool = False) -> str:
#     if use_alpha:
#         return f"I will provide you with {num} passages, each indicated by a alphabetical identifier []. Read all passages carefully and memorize them based on their relevance to the search query: {query}"
#     else:
#         return f"I will provide you with {num} passages, each indicated by a numerical identifier []. Read all passages carefully and memorize them based on their relevance to the search query: {query}"
#
# def _add_post_prompt(use_alpha, query: str, num: int, variable_passages: bool = False, rank1=7, rank2=8) -> str:
#     id1 = chr(ALPH_START_IDX + rank1) if use_alpha else str(rank1)
#     id2 = chr(ALPH_START_IDX + rank2) if use_alpha else str(rank2)
#     cand1, cand2 = f"[{id1}]", f"[{id2}]"
#     return f"Search Query: {query}.\nYou task is to compare the two documents I mention at the end. Only respond with the identifier, do not say any other word or explain. Now, compare which document is more relevant: Document {cand1} or Document {cand2}?"


def _add_few_shot_examples(conv, _examples=None, _num_few_shot_examples=0):
    for _ in range(_num_few_shot_examples):
        ex = random.choice(_examples)
        obj = json.loads(ex)
        prompt = obj["conversations"][0]["value"]
        response = obj["conversations"][1]["value"]
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], response)
    return conv

def _add_few_shot_examples_messages(messages, _examples=None, _num_few_shot_examples=0):
    for _ in range(_num_few_shot_examples):
        ex = random.choice(_examples)
        obj = json.loads(ex)
        prompt = obj["conversations"][0]["value"]
        response = obj["conversations"][1]["value"]
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": response})
    return messages


# Document identifier probability
# Binary classification probabilty

# def _add_prefix_prompt_doc_string(self, use_alpha, query: str, num: int) -> str:
# def _add_prefix_prompt_github_issue(self, use_alpha, query: str, num: int) -> str:
# def _add_post_prompt_github_issue(self, use_alpha, query: str, num: int) -> str:
# def _add_post_prompt_doc_string(self, use_alpha, query: str, num: int) -> str:

