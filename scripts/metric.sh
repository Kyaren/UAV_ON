cd "$(dirname "$0")/.."
root_dir=.
echo $PWD

python -u $root_dir/utils/classify_metric.py \
    --base_path $root_dir/logs
