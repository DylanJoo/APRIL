## Modularized LLM-Reranking  


## Strcutre
```
APRIL/
├── README.md # Project overview
├── requirements.txt # Python dependencies
└── src

src/
├── data/
│ └── processed/
│ 
├── examples/
│ └── exploration.ipynb
│ 
├── config_manager.py (see default_config.yaml)
│ 
├── reranking/result_parser
│ ├── base.py (abstract class)
│ └── _rank_gpt.py
│ 
├── reranking/prompt_builder
│ ├── base.py (abstract class)
│ └── _rank_gpt.py
│ 
├── reranking/llm_provider
│ ├── base.py (abstract class)
│ └── _rank_gpt.py
│ 
├── utils.py # Utility functions
│ 
├── tests/ # Unit tests 
└── test_model.py
```
#### Usage 
```python    
python -m reranking.wrapper \
    --config=reranking/configs/default_config.yaml \
    --data.shuffle=False 
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

