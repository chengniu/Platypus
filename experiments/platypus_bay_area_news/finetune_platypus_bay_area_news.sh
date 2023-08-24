set -x
set -e

export PYTHONPATH=$PYTHONPATH:/home/dayong/chengniu/Platypus

export CUDA_VISIBLE_DEVICES=0

cd /home/dayong/chengniu/Platypus

torchrun --nproc_per_node=1 --master_port=1234 \
finetune.py \
--base_model \
/home/dayong/chengniu/model/llama2_7B_hf \
--data-path \
/home/dayong/chengniu/data/platypus_bay_area_news.json \
--output_dir \
/home/dayong/chengniu/model/platypus_bay_area_news \
--batch_size \
16 \
--micro_batch_size \
1 \
--num_epochs \
1 \
--learning_rate \
0.0004 \
--cutoff_len \
4096 \
--val_set_size \
0 \
--lora_r \
16 \
--lora_alpha \
16 \
--lora_dropout \
0.05 \
--lora_target_modules \
'[gate_proj, down_proj, up_proj]' \
--train_on_inputs \
False \
--add_eos_token \
False \
--group_by_length \
False \
--prompt_template_name \
alpaca \
--lr_scheduler \
cosine \
--warmup_steps \
100 \
--gradient_checkpointing \
False \
--load_in_8bit \
True \

cd /home/dayong/chengniu/Platypus/experiments/platypus_bay_area_news
