## Modularized LLM-Reranking Library: 
This modularized LLM-Reranking project provides a flexible framework for reranking search results using large language models (LLMs). 
It allows users to easily experiment and integrate different components from different neural reranking methods. 

[July 4, 2025] We consider `sorting algorithm`, `prompting template` and corresponding `rankign parser` with different `LLM backend`.

## Strcutre
```
APRIL/ # the proposed new method using `reranking`.
├── pyproject.toml
├── README.md
├── .gitignore
├── unittest/li_textlist.py
├── src/
│   └── reranking/
│       ├── __init__.py
│       ├── config_manager.py
│       ├── utils.py
│       ├── prompt_builder/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── _rank_gpt.py
│       ├── llm_provider/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── _rank_gpt.py
│       ├── result_parser/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── _rank_gpt.py
│       └── tests/
│           ├── __init__.py
│           └── test_model.py
└── examples/
    └── exploration.ipynb
```

#### Wrapper/main functions
- ModularReranker (also the main function)
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

