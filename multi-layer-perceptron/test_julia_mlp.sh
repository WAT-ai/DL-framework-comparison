# 10 MLP trials
declare -a seeds=(42 0 100 87 1234 7 10 13 17 19)
declare -a frameworks=("flux" "knet")

for seed in ${seeds[@]}; do
    for framework in ${frameworks[@]}; do
        tsp julia ${framework}_test.jl --seed=$seed
    done
done