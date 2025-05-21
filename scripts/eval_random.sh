cd "$(dirname "$0")/.."

root_dir=.

echo $PWD
CUDA_VISIBLE_DEVICES=0,2,3 python -u $root_dir/src/eval_copy.py \
    --maxActions 150 \
    --save_path $root_dir/logs/eval_random \
    --eval_save_path $root_dir/logs/eval_random \
    --dataset_path ../DATA/TestSeen.json \
    --is_fixed  true\
    --gpu_id 1\
    --batchSize 6\
    --simulator_tool_port 31000\
    --generation_model_path Qwen/Qwen2.5-VL-7B-Instruct

