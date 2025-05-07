cd "$(dirname "$0")/.."

root_dir=.

echo $PWD
CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/eval.py \
    --maxActions 150 \
    --save_path $root_dir/logs/eval \
    --dataset_path /home/syx/Desktop/ModularNeighborhood/TestEpisode/UnSeenThings.json \
    --is_fixed  true\
    --gpu_id 0\
    --batchSize 2\

