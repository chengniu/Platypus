set -x
set -e

export PYTHONPATH=$PYTHONPATH:/home/dayong/chengniu/Platypus

cd /home/dayong/chengniu/Platypus

python merge.py \
--base_model_name_or_path /home/dayong/chengniu/model/llama2_7B_hf \
--peft_model_path /home/dayong/chengniu/model/platypus_bay_area_news_2 \
--output_dir /home/dayong/chengniu/model/platypus_bay_area_news_2_merged \

cd /home/dayong/chengniu/Platypus/experiments/platypus_bay_area_news_2
