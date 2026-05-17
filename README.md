## Modularized LLM-Reranking Library: 
This modularized LLM-Reranking project provides a flexible framework for reranking search results using large language models (LLMs). 
It allows users to easily experiment and integrate different components from different neural reranking methods. 

[July 4, 2025] We consider `sorting algorithm`, `prompting template` and corresponding `rankign parser` with different `LLM backend`.

## Installation
- Environment
```
conda create -n rerank python=3.10
pip install uv
```
- Dependency Installation
```
git clone https://github.com/DylanJoo/APRIL.git
cd APRIL
uv pip install -e .
uv pip install vllm==0.11.1 ftfy ir_datasets ir_measures 
```
The final `requirements.txt` will be provided in the future release.

## Strcutre [TODO: update to the beta version]
```
APRIL/ # the proposed new method using `autollmrerank`.
├── unittest/li_textlist.py
├── src/
│   └── autollmrerank/
│       ├── __init__.py
│       ├── config_manager.py
│       ├── utils.py
│       ├── prompt_builder/
│       │   ├── base.py
│       │   └── _rank_gpt.py
│       ├── llm_provider/
│       │   ├── base.py
│       │   └── _rank_gpt.py
│       ├── result_parser/
│       │   ├── base.py
│       │   └── _rank_gpt.py
│       └── tests/
│           └── test_model.py
└── examples/
    ├── run_trec-dl-2020.sh
    └── log.vllm/
```

#### Wrapper/main functions
- ModularReranker 
    * A wrapper class that defines the reranking types for class factory, which can integrates all the following 4 components.

#### Four modules 
- InputAssebler (rename? scheduler? handler? ...)
    * Input: query and results
    * Output: list of query-documents pairs

- PromptBuilder
    * Input: query and documents 
    * Output: text prompts for LLM

- LLMProvider
    * Input: text prompts
    * output: text outputs or list of numbers

- ResultParser
    * Input: text outputs or list of numbers
    * Output: Result object with sorted results

#### Utililty functions/classes
- Result: the class of retrieval/ranking results.
- PromptMode: the class of reranking mode, including the prompt, llm calling and parsing

#### Top 100 pool
Average pool size 255.1627906976744
Average pool size 145.625
Average pool size 325.32
Average pool size 278.23839009287923
Average pool size 278.408
Average pool size 291.54
Average pool size 264.40816326530614

#### Top 20 pool
Average pool size 97.90697674418605
Average pool size 41.815
Average pool size 144.32
Average pool size 122.30030959752322
Average pool size 116.86
Average pool size 123.94
Average pool size 127.40816326530613
