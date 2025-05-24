cd "$(dirname "$0")/.."

root_dir=.


echo $PWD
CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/eval_cliph.py \
    --maxActions 10 \
    --eval_save_path $root_dir/CLIP_logs/Neighborhood \
    --dataset_path ../DATA_test/Neighborhood2.json \
    --is_fixed  true\
    --gpu_id 0 \
    --batchSize 1 \
    --simulator_tool_port 32000

