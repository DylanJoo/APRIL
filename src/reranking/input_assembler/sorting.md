# Sorting algoritms 
<!-- TODO: maybe remove _topk and differentiate it with other vs. _all.-->

| Method        | Algorithm       | Type       | Prompt Builder | Post Processing   | Result Parser  |
| ---           | ---             | ---        | ---            | ---               | ---            |
| [x] RankGPT   | Bubble Sort     | listwise   |                |                   |                |
| [x] PairALL   |                 | pairwise   | Pairwise       | Sym-Sum           | Binary-prob    |
| [x] PairTopK  | Bubble TopK     | pairwise   | Pairwise       | bool(Sym-Sum > 0) | Binary-prob    |
| [ ] SetTopK   | Bubble TopK     | setwise    | Setwise        |                   | Dist-LogP      |
| [ ] PairTopK  | Quick TopK      | pairwise   |                |                   |                |
| [ ] SetTour   | Tournment Sort  | setwise    |                |                   |                |
| [ ] RefRank   | Reference       | Pointwise  |                |                   |                |


# To be implemented

| Method          | Type      | Algorithm        | Prompt Builder   | Result Parser  |
| ---             | ---       | ---              | ---              | ---            |
| [x] RankZephyr  | listwise  | Bubble Sort      |                  |                |
| [x] RankFIRST   | listwise  | Bubble Sort      |                  |                |
| [x] APRIL       | listwise  | Partition Sort   |                  |                |
