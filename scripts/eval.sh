cd "$(dirname "$0")/.."

root_dir=.


echo $PWD
CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/eval.py \
    --maxActions 150 \
    --eval_save_path $root_dir/logs/WinterTown \
    --dataset_path ../DATA_test/WinterTown.json \
    --is_fixed  true\
    --gpu_id 0 \
    --batchSize 1 \
    --simulator_tool_port 30000

