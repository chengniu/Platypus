set -x #echo on
set -e #stop when any command fails

export PYTHONPATH=$PYTHONPATH:/home/dayong/chengniu/Platypus

export CUDA_VISIBLE_DEVICES=0

python inference.py \
    --base_model garage-bAInd/Platypus2-7B \
    --lora_weights null \
    --csv_path /home/dayong/chengniu/data/bay_area_en_retrieved_gen.train.alpache.short.json \
    --output_csv_path /home/dayong/chengniu/results/Platypus2-7B/bay_area_en_retrieved_gen.train.alpache.out.csv
