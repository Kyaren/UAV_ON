cd "$(dirname "$0")/.."

root_dir=.


echo $PWD
CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/eval_3.py \
    --maxActions 150 \
    --eval_save_path $root_dir/logs/CityPark \
    --dataset_path ../DATA_R/CityPark_1.json \
    --is_fixed  true\
    --gpu_id 0 \
    --batchSize 1 \
    --simulator_tool_port 32000

