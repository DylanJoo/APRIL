import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class SlidingWindow:
    def __init__(
        self, 
        model_name: str, 
        device: str = 'cuda', 
        dtype=torch.float16
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto" if device == 'cuda' else None
        )
        self.device = device

    # from llmranker
    def sliding_windows_batched(
        self,
        retrieved_results: List[Result],
        use_logits: bool,
        use_alpha: bool,
        rank_start: int,
        rank_end: int,
        window_size: int,
        step: int,
        logging: bool = False,
    ) -> List[Result]:

        rerank_results = [copy.deepcopy(result) for result in retrieved_results]

        end_pos = rank_end
        start_pos = rank_end - window_size
        
        while end_pos > rank_start and start_pos + step != rank_start:
            start_pos = max(start_pos, rank_start)
            rerank_results = self.permutation_pipeline_batched(
                rerank_results, use_logits, use_alpha, start_pos, end_pos, logging
            )
            end_pos = end_pos - step
            start_pos = start_pos - step
        return rerank_results

    def permutation_pipeline_batched(
        self,
        results: List[Result],
        use_logits: bool,
        use_alpha: bool,
        rank_start: int,
        rank_end: int,
        logging: bool = False,
    ) -> List[Result]:

        prompts = self.create_prompt_batched(results, use_alpha, rank_start, rank_end, batch_size=32)
        batched_cache = self._llm.inference([prompt for (prompt, _) in prompts], use_logits=use_logits)

        # get the pairwise scores
        batch_score_matrix = torch.zeors( (len(results), rank_end-rank_start, rank_end-rank_start) )
        batched_results = self.run_llm_pair_batched(
            results, 
            batched_cache, 
            score_matrix=batch_score_matrix,
            aggregation_type='sum', 
            use_logits=use_logits, use_alpha=use_alpha, current_window_size=rank_end - rank_start
        )

        for index, (result, (prompt, _)) in enumerate(zip(results, prompts)):
            permutation, out_token_count = batched_results[index]
            result.ranking_exec_summary = prompt
            result = self.receive_permutation(result, permutation, rank_start, rank_end, use_alpha)

        return results

    def build_listwise_prompt_batch(self, queries: List[str], passages_list: List[List[str]]) -> List[str]:
        prompts = []
        def build(qp):
            q, p = qp
            return self.build_listwise_prompt(q, p)

        with ThreadPoolExecutor() as executor:
            for prompt in tqdm(executor.map(build, zip(queries, passages_list)), total=len(queries), desc="Building prompts"):
                prompts.append(prompt)

        return prompts

    def preload_static_cache_batch(self, queries: List[str], passages_list: List[List[str]]):
        prompts = self.build_listwise_prompt_batch(queries, passages_list)
        static_inputs = self.tokenizer(prompts, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**static_inputs, use_cache=True)
        return outputs.past_key_values

    def build_pairwise_prompts(self, pairs: List[Tuple[str, str]]) -> List[str]:
        prompts = [f"Compare passage {a} and passage {b} based on the query." for a, b in pairs]
        return prompts

    def batch_pairwise_inference_multiquery(self, all_pairs: List[List[Tuple[str, str]]], static_cache, batch_size: int = 32) -> List[List[str]]:
        batch_prompts = []
        query_mapping = []
        for query_idx, pairs in enumerate(all_pairs):
            prompts = self.build_pairwise_prompts(pairs)
            batch_prompts.extend(prompts)
            query_mapping.extend([query_idx] * len(prompts))

        results = [[] for _ in range(len(all_pairs))]

        for i in range(0, len(batch_prompts), batch_size):
            batch_inputs = self.tokenizer(batch_prompts[i:i+batch_size], return_tensors='pt', padding=True).to(self.device)

            past = tuple(layer[:, [query_mapping[j+i] for j in range(batch_inputs.input_ids.size(0))], :, :]
                         for layer in static_cache)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch_inputs.input_ids,
                    attention_mask=batch_inputs.attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    max_new_tokens=1,
                )

            logits = outputs.logits[:, -1, :]
            predictions = torch.argmax(logits, dim=-1)
            tokens = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

            for j, token in enumerate(tokens):
                results[query_mapping[i+j]].append(token)

        return results

    def rank_passages_with_sliding(self, query: str, passages: List[str], window_size: int = 10, step_size: int = 5, batch_size: int = 32) -> List[int]:
        global_win_counts = {i: 0 for i in range(len(passages))}

        queries, passages_list, all_pairs = [], [], []
        start_indices = []

        for start_idx in range(0, len(passages) - window_size + 1, step_size):
            window_passages = passages[start_idx:start_idx+window_size]
            labels = [chr(ord('A') + i) for i in range(len(window_passages))]
            pairs = [(a, b) for idx, a in enumerate(labels) for b in labels[idx+1:]]

            queries.append(query)
            passages_list.append(window_passages)
            all_pairs.append(pairs)
            start_indices.append(start_idx)

        static_cache = self.preload_static_cache_batch(queries, passages_list)
        all_results = self.batch_pairwise_inference_multiquery(all_pairs, static_cache, batch_size=batch_size)

        for (pairs, comparison_results, start_idx) in zip(all_pairs, all_results, start_indices):
            labels = [chr(ord('A') + i) for i in range(len(passages[start_idx:start_idx+window_size]))]
            for (a, b), result in zip(pairs, comparison_results):
                local_winner = a if 'A' in result or 'a' in result else b
                local_idx = labels.index(local_winner)
                global_idx = start_idx + local_idx
                if global_idx < len(passages):
                    global_win_counts[global_idx] += 1

        ranking = sorted(global_win_counts.items(), key=lambda x: -x[1])
        ranking_indices = [idx for idx, _ in ranking]
        return ranking_indices

    def create_prompt(
        query: str,
        result: Result,
        use_alpha: bool, 
        rank_start: int,
        rank_end: int,
    ) -> Tuple[Tuple(str, str), int]:

        query = result.query
        num_passages = len(result.hits)
        max_length = 300

        while True:
            messages = list()
            if self._system_message and self.system_message_supported:
                messages.append({"role": "system", "content": self._system_message})

            rank = 0
            input_context = f"""I will provide you with {num_passages} passages, each indicated by a alphabetical identifier []. Read and memorize all passages carefully. Your will use these passages for multiple comparisons based on their relevance to the search query: {query}\n\n"""
            for hit in result.hits[rank_start:rank_end]:
                rank += 1
                content = hit['content'].replace("Title: Content", "").strip()
                content = " ".join(content.split()[:max_length])
                identifier = chr(ALPH_START_IDX + rank) if use_alpha else str(rank)
                input_context += f"[{identifier}] {content}\n"

            input_context += f"""\nSearch Query: {query}\nBased on the search query, focus on comparing the passages:[cache_input_split][identifier_cand1] and [identifier_cand2]. Respond only with the identifier of the passage that is more relevant."""
            messages.append({"role": "user", "content": input_context})

            # will only use the first message
            if self._system_message and not self.system_message_supported:
                messages[0]["content"] = self._system_message + "\n " + messages[0]["content"]

            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt = fix_text(prompt)
            prompt = prompt.split("[cache_input_split]")

        return prompt, 0

    def create_prompt_batched(
        self,
        results: List[Result],
        use_alpha: bool,
        rank_start: int,
        rank_end: int,
        batch_size: int = 32,
    ) -> List[Tuple[str, int]]:
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        all_completed_prompts = []

        with ThreadPoolExecutor() as executor:
            for batch in tqdm(chunks(results, batch_size), desc="Processing batches"):
                completed_prompts = list(
                    executor.map(
                        lambda result: self.create_prompt(result, use_alpha, rank_start, rank_end),
                        batch,
                    )
                )
                all_completed_prompts.extend(completed_prompts)
        return all_completed_prompts

# Example usage
if __name__ == "__main__":
    model_name = "gpt2"  # Replace with your model
    ranker = RankLLM(model_name)

    query_passages = {
        "How to manage mild asthma symptoms?": [
            "Use a rescue inhaler as needed.",
            "Consider long-term corticosteroid inhalers.",
            "Identify and avoid asthma triggers.",
            "Monitor symptoms regularly.",
            "Have an action plan for asthma attacks.",
            "Stay physically active to strengthen lungs.",
            "Take prescribed medications daily.",
            "Schedule regular doctor visits.",
            "Consider allergy treatments.",
            "Practice breathing exercises."
        ],
        "Best ways to prevent seasonal allergies?": [
            "Stay indoors during high pollen counts.",
            "Use air purifiers at home.",
            "Shower after outdoor activities.",
            "Wear masks when outdoors.",
            "Take antihistamines before symptoms start.",
            "Keep windows closed.",
            "Monitor weather reports.",
            "Consult an allergist for treatment.",
            "Avoid outdoor activities on windy days.",
            "Use nasal sprays to reduce symptoms."
        ]
    }

    rankings = ranker.rank_multiple_queries(query_passages, window_size=5, step_size=2, batch_size=4)
    for query, ranking in rankings.items():
        print(f"Query: {query}")
        print("Ranking order (passage indices):", ranking)
        print()

