"""
Example usage of HuggingFace dataset loader with the reranking pipeline.

This example demonstrates how to use load_hf to load queries and corpus
from HuggingFace datasets and integrate them with the reranking pipeline.
"""
import os
from pathlib import Path
from reranking import loader
from pprint import pprint

# Example: Using HuggingFace datasets instead of ir_datasets
def example_hf_loading():
    """
    Example of loading data from HuggingFace datasets.
    
    Query dataset structure:
        - Dataset: 'DylanJHJ/nano-beir'
        - Fields: 'query_id', 'query_texts'
        
    Corpus dataset structure:
        - Dataset: 'DylanJHJ/nano-beir-corpus'
        - Fields: 'docid', 'title', 'text'
    """
    
    # Load queries and corpus from HuggingFace datasets
    corpus, queries, qrels = loader.load_hf(
        queries_dataset_name='DylanJHJ/nano-beir',
        corpus_dataset_name='DylanJHJ/nano-beir-corpus',
        subset='nq',  # or 'msmarco', 'fiqa', etc.
        query_split='test',
        corpus_split='train',
    )
    
    print(f"Loaded {len(queries)} queries")
    print(f"Loaded {len(corpus)} documents")
    print(f"\nExample query: {list(queries.items())[0]}")
    print(f"Example document: {list(corpus.items())[0]}")
    
    return corpus, queries, qrels


def example_with_reranking():
    """
    Complete example showing integration with the reranking pipeline.
    
    Note: This requires a run file and actual model setup.
    """
    from reranking.config_manager import ConfigManager
    from reranking.wrapper import AutoLLMReranker
    
    # Initialize the reranker configuration
    config = ConfigManager(
        rerank_mode='RankGPT',
        top_k=100,
        rank_start=0,
        rank_end=100,
        window_size=20,
        num_runs=1,
        llm={'max_model_len': 8196, 'model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct'}
    ).get_config()
    
    # Load data from HuggingFace
    corpus, queries, qrels = loader.load_hf(
        queries_dataset_name='DylanJHJ/nano-beir',
        corpus_dataset_name='DylanJHJ/nano-beir-corpus',
        subset='nq',
        query_split='test',
        corpus_split='train',
    )
    
    # Load initial run (BM25 or other baseline results)
    # run = loader.load_run(config.data.input_run)
    # Or create a dummy run for testing:
    run = {}
    for qid in list(queries.keys())[:5]:  # Just use first 5 queries
        run[qid] = {}
        for i, docid in enumerate(list(corpus.keys())[:20]):  # Top 20 docs
            run[qid][docid] = 1.0 / (i + 1)  # Simple score
    
    # Initialize the reranker
    rankllm = AutoLLMReranker(
        config, 
        system_message="You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query"
    )
    
    # Perform reranking
    reranked_run = rankllm.rerank(
        run=run,
        queries=queries,
        corpus=corpus,
        query_batch_size=2,
    )
    
    print(f"\nReranked {len(reranked_run)} queries")
    return reranked_run


def example_command_line_usage():
    """
    Example of using load_hf with command line wrapper.
    
    Instead of using ir_datasets, you would modify the wrapper script to use:
    
    ```python
    # In your script or wrapper.py __main__ section:
    
    # Option 1: Use load_hf instead of load
    corpus, queries, qrels = loader.load_hf(
        queries_dataset_name='DylanJHJ/nano-beir',
        corpus_dataset_name='DylanJHJ/nano-beir-corpus',
        subset='nq',
    )
    
    # Then use as normal:
    run = loader.load_run(config.data.input_run)
    reranked_run = rankllm.rerank(run=run, queries=queries, corpus=corpus)
    ```
    
    Or create a new script similar to existing unittest files but using load_hf.
    """
    pass


if __name__ == "__main__":
    print("=" * 70)
    print("Example 1: Loading from HuggingFace datasets")
    print("=" * 70)
    try:
        corpus, queries, qrels = example_hf_loading()
    except Exception as e:
        print(f"Note: Example failed because datasets are not accessible: {e}")
        print("This is expected if the datasets are private or not yet published.")
    
    print("\n" + "=" * 70)
    print("Example 2: Integration with reranking (requires model setup)")
    print("=" * 70)
    print("See function example_with_reranking() for implementation details")
    
    print("\n" + "=" * 70)
    print("Example 3: Command line usage pattern")
    print("=" * 70)
    print("See function example_command_line_usage() for usage pattern")
