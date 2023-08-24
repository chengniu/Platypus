set -x
set -e

export PYTHONPATH=$PYTHONPATH:/home/dayong/chengniu/lm-evaluation-harness

export CUDA_VISIBLE_DEVICES=0

cd /home/dayong/chengniu/lm-evaluation-harness

python main.py --model hf-causal-experimental --model_args pretrained=/home/dayong/chengniu/model/platypus_bay_area_news_merged,use_accelerate=True,dtype="bfloat16" --tasks arc_challenge --batch_size 2 --no_cache --write_out --output_path /home/dayong/chengniu/results/platypus_bay_area_news_merged/arc_challenge_25shot.json --device cuda --num_fewshot 25

python main.py --model hf-causal-experimental --model_args pretrained=/home/dayong/chengniu/model/platypus_bay_area_news_merged,use_accelerate=True,dtype="bfloat16" --tasks hellaswag --batch_size 2 --no_cache --write_out --output_path /home/dayong/chengniu/results/platypus_bay_area_news_merged/hellaswag_10shot.json --device cuda --num_fewshot 10

python main.py --model hf-causal-experimental --model_args pretrained=/home/dayong/chengniu/model/platypus_bay_area_news_merged,use_accelerate=True,dtype="bfloat16" --tasks hendrycksTest-* --batch_size 2 --no_cache --write_out --output_path /home/dayong/chengniu/results/platypus_bay_area_news_merged/mmlu_5shot.json --device cuda --num_fewshot 5

python main.py --model hf-causal-experimental --model_args pretrained=/home/dayong/chengniu/model/platypus_bay_area_news_merged,use_accelerate=True,dtype="bfloat16" --tasks truthfulqa_mc --batch_size 2 --no_cache --write_out --output_path /home/dayong/chengniu/results/platypus_bay_area_news_merged/truthfulqa_0shot.json --device cuda

cd /home/dayong/chengniu/Platypus/experiments/platypus_bay_area_news
