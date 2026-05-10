for file in *rerank*;do
    for set in $file/*;do
        for eval_file in $set/*;do
            count=$(cat $eval_file | wc -l)
            if [ "$count" -ne 8 ]; then
                echo $file $set $eval_file
                echo $count
            fi
        done
    done
done
