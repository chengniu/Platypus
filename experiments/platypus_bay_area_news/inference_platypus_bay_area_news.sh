set -x
set -e

export PYTHONPATH=$PYTHONPATH:/home/dayong/chengniu/Platypus

cd /home/dayong/chengniu/Platypus

export CUDA_VISIBLE_DEVICES=0

python inference.py \
--base_model /home/dayong/chengniu/model/platypus_bay_area_news_merged \
--lora_weights null \
--input_csv_path /home/dayong/chengniu/data/20230815_week1.alpache.short.json \
--output_csv_path /home/dayong/chengniu/results/platypus_bay_area_news_merged/20230815_week1.alpache.short.txt \
--output_benchmark_path /home/dayong/chengniu/benchmark \
--model_name platypus_bay_area_news_merged \
--output_qa True \

cd /home/dayong/chengniu/Platypus/experiments/platypus_bay_area_news
