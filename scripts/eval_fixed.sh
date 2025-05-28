cd "$(dirname "$0")/.."
root_dir=.
echo $PWD

scenes = {"Barnyard", "BrushifyRoad", "BrushifyUrban", "CabinLake", "CityPark", "CityStreet", "DownTown", \
        "Neighborhood", "NYC", "Slum", "UrbanJapan", "Venice", "WesternTown", "WinterTown"}

for scene in "${scenes[@]}"
do 
    echo "Evaluating scene: $scene"
    CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/eval_2.py \
        --maxActions 150 \
        --eval_save_path $root_dir/logs/"${scene}" \
        --dataset_path your/dataset/path \
        --is_fixed true \
        --gpu_id 0 \
        --batchSize 1 \
        --simulator_tool_port 30000
done

### alse can eval every scene separately

# CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/eval_2.py \
#     --maxActions 150 \
#     --eval_save_path $root_dir/logs/scene \
#     --dataset_path your/dataset/path \
#     --is_fixed  true\
#     --gpu_id 0 \
#     --batchSize 1 \
#     --simulator_tool_port 30000
