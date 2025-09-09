<h3>Example Usage for Python AutoReranker</h3>
---

# Example data
To accomondate to the standard input format of AutoReranker, the example data is organized as three dictionaries: `run`, `queries`, and `corpus`.
Below is an example of how to structure these dictionaries.
```python
run = {
    "q1": {"d2": 0.95, "d1*": 0.70, "d6": 0.25},
    "q2": {"d4*": 0.88, "d3": 0.73, "d7": 0.20},
    "q3": {"d5*": 0.91, "d8": 0.60, "d9*": 0.40}
}

queries = {
    "q1": "What city is the capital of France?",
    "q2": "Who painted the Mona Lisa?",
    "q3": "√144 equals?"
}

corpus = {
    "d1*": "Paris is the capital of France.",
    "d2": "London is the capital of the UK.",
    "d3": "Vincent van Gogh painted 'The Starry Night'.",
    "d4*": "The painter of the Mona Lisa was Leonardo da Vinci.",
    "d5*": "The square root of 144 is 12.",
    "d6": "Berlin is the capital of Germany.",
    "d7": "Pablo Picasso painted 'Guernica'.",
    "d8": "The cube root of 27 is 3.",
    "d9*": "12 is the positive solution to √144."
}
```

# Initialize a reranker and rerank
Once the data is structured, you can initialize the `ModularReranker` with the prebuilt method and use it to rerank the documents based on the queries.
```python
reranker = ModularReranker.from_prebuilt('rankgpt', 'Qwen/Qwen2.5-7B-Instruct')
result = reranker.rerank(run=run, queries=queries, corpus=corpus)
```
