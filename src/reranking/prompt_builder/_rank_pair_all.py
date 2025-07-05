import re
from typing import List, Optional, Union, Callable, Dict, Tuple

class RankPairAllFormatter:
    def __init__(
        self, 
        use_alpha=False, 
        variable_passages=False,
        max_doc_length=1024
    ):
        self._use_alpha = use_alpha
        self._variable_passages = variable_passages 

        if use_alpha: 
            self.id_type = "alphabetical"
            self.example_ordering = "[B] > [A]" if variable_passages else "[D] > [B]"
        else:
            self.id_type = "numerical"
            self.example_ordering = "[2] > [1]" if variable_passages else "[4] > [2]"

        self.max_doc_length = max_doc_length

    # [TODO] Equalize the max length
    def _document_format(self, doc: Union[str, Dict]) -> str:
        if isinstance(doc, dict):
            title = doc.get('title', False)
            if 'contents' in doc:
                text = doc['contents'].strip()
                text = f"Title: {title} Content: {text}" if title else text
            else:
                raise ValueError(f"Incorrect document dictionary format. Expected keys: 'title', 'contents': got {doc}")
        elif isinstance(doc, str):
            text = doc.strip()
        else:
            raise ValueError(f"Document must be a string or a dictionary with 'content' key: got {doc}")

        return " ".join(text.split()[:self.max_doc_length])  

    def prefix(self, query: str, doc_list: Optional[List[Dict]] = None, **kwargs) -> str:
        template = (
            f"I will provide you with two passages. Read and memorize both carefully. "
            f"Your task is to determine which passage is more relevant to the query.\n\n"
            f"Query: {query}\n\n"
            f"Passage 1: {doc1}\n"
            f"Passage 2: {doc2}\n\n"
            f"Based on the given query, is Passage 1 more relevant than Passage 2? "
            f"Please answer 'Yes' or 'No'.\nAnswer: "
        )

    def postfix(self, query: str, doc_list: Optional[List[Dict]] = None, **kwargs) -> str:
        return (
            f"Search Query: {query}.\n"
            f"Rank the {len(doc_list)} passages above based on their relevance to the search query. "
            f"All the passages should be included and listed using identifiers, "
            f"in descending order of relevance. The output format should be [] > [], "
            f"e.g., {self.example_ordering}, "
            f"Only respond with the ranking results, do not say any word or explain."
        )

    def body(self, query: str, doc_list: Optional[List[Dict]], **kwargs) -> str:
        prompt_body = ""
        # for i, doc in enumerate(doc_list, start=1): # chr(65) is 'A'
        #     identifier = f"[{chr(64 + i)}]" if self._use_alpha else f"[{i}]"
        #     doc_text = self._document_format(doc)
        #     prompt_body += f"{identifier} {self.replace_number(doc_text)}\n"
        # return prompt_body

        # Enumerate
        idx_pairs = [(i, j) for i in range(len(doc_list)) for j in range(len(doc_list)) if i != j]
        prompts = []

        for i, j in idx_pairs:
            prompt = self.template.format(query=query doc1=doc_list[i], doc2=doc_list[j])
            prompts.append(prompt)
            id_pairs.append((qid, i, j))

        return "\n".join(prompts)

    def replace_number(self, text: str) -> str:
        if self._use_alpha:
            return re.sub(r"\[([A-z]+)\]", r"(\1)", text)
        else:
            return re.sub(r"\[(\d+)\]", r"(\1)", text)


# # Build pairwise prompts
# for result in init_results:
# qid = result.qid
# docids = [doc.docid for doc in result.hits]
# query = result.query
# pairs = [(i, j) for i in range(len(docids)) for j in range(len(docids)) if i != j]
#
# for i, j in pairs:
#     prompt = self.template.format(
#         cand1=result.doc_map[docids[i]]["contents"],
#         cand2=result.doc_map[docids[j]]["contents"],
#         query=query
#     )
#     prompts.append(prompt)
#     id_pairs.append((qid, i, j))
#
# logger.info(f"Number of prompts: {len(prompts)}")
#
# # Set up tokenizer and label mapping
# true_list = [' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES']
# false_list = [' No', 'No', ' no', 'no', 'NO', ' NO']
# self._llm.set_classification(true_list, false_list)

