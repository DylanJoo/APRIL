for judge in ${HOME}/APRIL/qrel-analysis/dense-7b/*;do
    echo $judge
    collected_path=${HOME}/APRIL/qrel-analysis/dense-7b/${judge##*/}/all.jsonl
    cat $judge/*/* > $collected_path
done

# for judge in ${HOME}/APRIL/qrel-analysis/dense-7b/*rankzephyr*;do
#     rmdir $judge
#     rm $judge/all.jsonl
#     for j in $judge/*;do
#         rmdir $j
#     done
# done
