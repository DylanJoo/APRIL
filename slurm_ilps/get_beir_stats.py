from autollmrerank.loader_dev.irds import load

BEIR_DATASETS = \
["arguana", "climate-fever", "dbpedia-entity/test", "fever/test", "fiqa/test", "hotpotqa/test", "nfcorpus/test", "nq", "quora/test", "scidocs", "scifact/test", "trec-covid", "webis-touche2020/v2"]

for dataset in BEIR_DATASETS:
    corpus, queries, qrels = load(f"beir/{dataset}")

    print("Dataset | # Queries | # Corpus | # Judgments |" + \
          "(Avg) Q length | (Avg) D length | (Avg) d+/Q | (Avg) d-/Q")

    ## Total numbers
    num_queries = len(queries)
    num_corpus = len(corpus)
    num_judgments = [len(qrels[qid]) for qid in qrels]

    ## Average numbers
    avg_q_length = sum([len(queries[qid].split()) for qid in queries]) / num_queries
    avg_d_length = sum([len(corpus[did]['contents'].split()) for did in corpus]) / num_corpus

    ## Calcuate the positive and negative
    avg_d_plus_q = {}
    avg_d_minus_q = {}
    for qid in qrels:
        num_d_plus = sum([1 for did in qrels[qid] if qrels[qid][did] > 0])
        num_d_minus = sum([1 for did in qrels[qid] if qrels[qid][did] <= 0])
        avg_d_plus_q[qid] = num_d_plus
        avg_d_minus_q[qid] = num_d_minus

    avg_d_plus_q = sum(avg_d_plus_q.values()) / num_queries
    avg_d_minus_q = sum(avg_d_minus_q.values()) / num_queries
    avg_d_q = avg_d_plus_q + avg_d_minus_q


    ## print out
    print(f"{dataset} | {num_queries} | {num_corpus} | {sum(num_judgments)} | " + \
          f"{avg_q_length:.2f} | {avg_d_length:.2f} | {avg_d_plus_q:.2f} | {avg_d_minus_q:.2f}")
