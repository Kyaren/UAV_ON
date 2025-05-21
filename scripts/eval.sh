cd "$(dirname "$0")/.."

root_dir=.


echo $PWD
CUDA_VISIBLE_DEVICES=3,4,5 python -u $root_dir/src/eval.py \
    --maxActions 30 \
    --save_path $root_dir/logs/eval_constraint_Seen \
    --dataset_path ../DATA/SeenThings.json \
    --is_fixed  true\
    --gpu_id 1 \
    --batchSize 6 \
    --simulator_tool_port 30000

