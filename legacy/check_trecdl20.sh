#!/bin/bash
result_file=runs/PairTopK/RankGPT/run.msmarco-v1-passage.bm25-trec-dl-2019.txt
label_file=${HOME}/.ir_datasets/msmarco-passage/trec-dl-2019/qrels

awk_labels=$(mktemp)

# Build a simplified lookup: key = qid_docid, value = label
awk '{print $1 "_" $3, $4}' "$label_file" > "$awk_labels"

awk -v label_lookup="$awk_labels" '
BEGIN {
    while ((getline < label_lookup) > 0) {
        label_map[$1] = $2
    }
}
{
    qid = $1
    docid = $3

    if (qid != last_qid) {
        count = 0
        last_qid = qid
    }

    if (count > 10) {
        key = qid "_" docid
        label = (key in label_map) ? label_map[key] : -1
        if (label == 3) {
            print qid, docid, count, label
        }
    }
    count++
}
' "$result_file"

rm "$awk_labels"
