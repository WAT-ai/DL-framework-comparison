# 10 MLP trials
declare -a seeds=(42 0 100 87 1234 7 10 13 17 19)
# declare -a seeds=(42)
declare -a frameworks=("jax" "mxnet" "pytorch" "tensorflow")

for seed in ${seeds[@]}; do
    for framework in ${frameworks[@]}; do
        tsp python3 ${framework}_test.py --seed=$seed
    done
done