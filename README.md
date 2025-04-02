# APRIL
Accelerating Pairwise re-ranking with listwise prompts

---
## Requirements

- Environment & 
```
conda create -n april python=3.10
conda activate april
conda install -c conda-forge openjdk=21 maven -y
```

## Prepare runs 
The top-1000 run files used in this work can be found in [runs](runs/). 
We follow [Pyserini 2cr](https://castorini.github.io/pyserini/2cr/msmarco-v1-passage.html) for the reproduction.

- BM25+RM3 (K1=0.9, b=0.4)
