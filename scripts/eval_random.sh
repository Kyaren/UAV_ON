cd "$(dirname "$0")/.."
root_dir=.

echo $PWD
CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/eval_random.py \
    --maxActions 150 \
    --eval_save_path $root_dir/random_logs/CityPark\
    --dataset_path ../DATA_test/CityPark.json \
    --is_fixed  true\
    --gpu_id 0\
    --batchSize 1\
    --simulator_tool_port 31000\
   
