from experiments.experiment_util import get_program_config
import os
import sys


def setup_experiments(env_name, fine_tune_name, platypus_src_path, lm_evaluation_src_path):
    _, env_config = get_program_config(env_name, platypus_src_path)
    finetune_args, finetune_config = get_program_config(fine_tune_name, platypus_src_path)
    output_dir = os.path.join(platypus_src_path, f'experiments/{fine_tune_name}')
    os.makedirs(output_dir, exist_ok=True)
    create_finetune_sh(fine_tune_name, env_config, finetune_args, finetune_config, platypus_src_path)
    create_evaluation_sh(fine_tune_name, env_config, finetune_args, finetune_config, platypus_src_path, lm_evaluation_src_path)



def create_finetune_sh(fine_tune_name, env_config, finetune_args, finetune_config, platypus_src_path):
    file_name = os.path.join(platypus_src_path, f"experiments/{fine_tune_name}/finetune_{fine_tune_name}.sh")
    with open(file_name, 'w', encoding='UTF8') as f:
        f.write(f"set -x\n")
        f.write(f"set -e\n")
        f.write('\n')
        f.write(f'export PYTHONPATH=$PYTHONPATH:{platypus_src_path}\n')
        f.write('\n')
        f.write(f'export CUDA_VISIBLE_DEVICES={env_config["--CUDA_VISIBLE_DEVICES"]}\n')
        f.write('\n')
        f.write(f'cd {platypus_src_path}\n')
        f.write('\n')
        nproc_per_node = env_config['--nproc_per_node']
        master_port = env_config['--master_port']
        f.write(f"torchrun --nproc_per_node={nproc_per_node} --master_port={master_port} \\\n")
        for one in finetune_args:
            f.write(f"{one} \\\n")
        f.write('\n')
        f.write(f'cd {platypus_src_path}/experiments/{fine_tune_name}\n')

    file_name = os.path.join(platypus_src_path, f"experiments/{fine_tune_name}/merge_{fine_tune_name}.sh")
    with open(file_name, 'w', encoding='UTF8') as f:
        f.write(f"set -x\n")
        f.write(f"set -e\n")
        f.write('\n')
        f.write(f'export PYTHONPATH=$PYTHONPATH:{platypus_src_path}\n')
        f.write('\n')
        f.write(f'cd {platypus_src_path}\n')
        f.write('\n')
        f.write("python merge.py \\n")
        f.write(f"--base_model_name_or_path {finetune_config['--base_model']} \\\n")
        f.write(f" --peft_model_path {finetune_config['--output_dir']} \\\n")
        f.write(f"--output_dir {finetune_config['--output_dir']}_merged \\\n")
        f.write('\n')
        f.write(f'cd {platypus_src_path}/experiments/{fine_tune_name}\n')


def create_evaluation_sh(fine_tune_name, env_config, finetune_args, finetune_config, platypus_src_path, lm_evaluation_src_path):
    model_path = f"{finetune_config['--output_dir']}_merged"
    result_path = env_config['--result_path']
    file_name = os.path.join(lm_evaluation_src_path, f"{platypus_src_path}/experiments/{fine_tune_name}/evaluate_{fine_tune_name}.sh")
    with open(file_name, "w", encoding="UTF8") as f:
        f.write(f"set -x\n")
        f.write(f"set -e\n")
        f.write('\n')
        f.write(f'export PYTHONPATH=$PYTHONPATH:{lm_evaluation_src_path}\n')
        f.write('\n')
        f.write(f'export CUDA_VISIBLE_DEVICES={env_config["--CUDA_VISIBLE_DEVICES"]}\n')
        f.write('\n')
        f.write(f'cd {lm_evaluation_src_path}\n')
        f.write('\n')
        f.write(f'python main.py --model hf-causal-experimental --model_args pretrained={model_path},use_accelerate=True,dtype="bfloat16" --tasks arc_challenge --batch_size 2 --no_cache --write_out --output_path {result_path}/{fine_tune_name}/arc_challenge_25shot.json --device cuda --num_fewshot 25\n')
        f.write('\n')
        f.write(f'python main.py --model hf-causal-experimental --model_args pretrained={model_path},use_accelerate=True,dtype="bfloat16" --tasks hellaswag --batch_size 2 --no_cache --write_out --output_path {result_path}/{fine_tune_name}/hellaswag_10shot.json --device cuda --num_fewshot 10\n')
        f.write('\n')
        f.write(f'python main.py --model hf-causal-experimental --model_args pretrained={model_path},use_accelerate=True,dtype="bfloat16" --tasks hendrycksTest-* --batch_size 2 --no_cache --write_out --output_path {result_path}/{fine_tune_name}/mmlu_5shot.json --device cuda --num_fewshot 5\n')
        f.write('\n')
        f.write(f'python main.py --model hf-causal-experimental --model_args pretrained={model_path},use_accelerate=True,dtype="bfloat16" --tasks truthfulqa_mc --batch_size 2 --no_cache --write_out --output_path {result_path}/{fine_tune_name}/truthfulqa_0shot.json --device cuda\n')
        f.write('\n')
        f.write(f'cd {platypus_src_path}/experiments/{fine_tune_name}\n')


if __name__ == "__main__":
    env_name = sys.argv[1]
    fine_tune_name = sys.argv[2]
    self_path = os.path.abspath(sys.argv[0])
    platypus_src_path = os.path.dirname(os.path.dirname(self_path))
    lm_evaluation_src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self_path))), "lm-evaluation-harness")

    setup_experiments(env_name, fine_tune_name, platypus_src_path, lm_evaluation_src_path)
