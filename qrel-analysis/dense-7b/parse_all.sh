for judge in ${HOME}/APRIL/qrel-analysis/dense-7b/*;do
    echo $judge
    collected_path=${HOME}/APRIL/qrel-analysis/dense-7b/${judge##*/}/all.jsonl
    cat $judge/*/* > $collected_path
done
