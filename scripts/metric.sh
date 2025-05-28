cd "$(dirname "$0")/.."
root_dir=.
echo $PWD

python -u $root_dir/utils/metric.py \
    --base_path $root_dir/logs