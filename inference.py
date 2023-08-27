import os
import sys
import time
import pandas as pd
import json
import fire
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from pathlib import Path
from utils.callbacks import Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def get_output_files(output_csv_path, output_benchmark_path, output_qa, model_name):
    os.makedirs(Path(output_csv_path).parent, exist_ok=True)
    output_csv_f = open(output_csv_path, "w", encoding='UTF8')
    if output_qa:
        os.makedirs(output_benchmark_path, exist_ok=True)
        question_f = open(os.path.join(output_benchmark_path, 'question.jsonl'), 'w', encoding='UTF8')
        answer_dir = os.path.join(output_benchmark_path, 'reference_answer')
        os.makedirs(answer_dir, exist_ok=True)
        answer_f = open(os.path.join(answer_dir, 'answer.jsonl'), 'w', encoding='UTF8')
    else:
        question_f = None
        answer_f = None
    model_dir = os.path.join(output_benchmark_path, 'model_answer')
    os.makedirs(model_dir, exist_ok=True)
    model_f = open(os.path.join(model_dir, f'{model_name}.jsonl'), 'w', encoding='UTF8')
    return output_csv_f, question_f, answer_f, model_f


def flush_output_files(out_f, q_f, a_f, m_f):
    for f in [out_f, q_f, a_f, m_f]:
        if f is not None:
            f.flush()
            os.fsync(f.fileno())


def close_output_files(out_f, q_f, a_f, m_f):
    for f in [out_f, q_f, a_f, m_f]:
        if f is not None:
            f.close()


def main(
    load_8bit: bool = True,
    base_model: str = "../llama30B_hf",
    lora_weights: str = "",
    prompt_template: str = "alpaca",
    input_csv_path: str = "",
    output_csv_path: str = "",
    output_benchmark_path: str = "",
    model_name: str = "",
    output_qa: bool = False
):
    out_f, q_f, a_f, m_f = get_output_files(output_csv_path, output_benchmark_path, output_qa, model_name)

    base_model = base_model or os.environ.get("BASE_MODEL", "")

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if lora_weights is not None and len(lora_weights) > 0 and lora_weights != 'null':
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.bfloat16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if input_csv_path.endswith("json") or input_csv_path.endswith('jsonl'):
        lines = []
        with open(input_csv_path) as f:
            lines = f.read().splitlines()
        line_dicts = [json.loads(line) for line in lines]
        df = pd.DataFrame(line_dicts)
    else:
        df = pd.read_csv(input_csv_path)
    instructions = df["instruction"].tolist()
    inputs = df["input"].tolist()
    answers = None
    try:
        answers = df['output'].tolist()
    except:
        answers = [None] * len(inputs)

    max_batch_size = 16

    for i in range(0, len(instructions), max_batch_size):
        instruction_batch = instructions[i:i + max_batch_size]
        input_batch = inputs[i:i + max_batch_size]
        answer_batch = answers[i:i + max_batch_size]
        print(f"Processing batch {i // max_batch_size + 1} of {len(instructions) // max_batch_size + 1}...")
        start_time = time.time()
    
        prompts = [prompter.generate_prompt(instruction, input) for instruction, input in zip(instruction_batch, input_batch)]
        evaluate(i // max_batch_size, prompter, prompts, answer_batch, model, tokenizer,
                 i * max_batch_size, model_name, out_f, q_f, a_f, m_f)
        print(f"Finished processing batch {i // max_batch_size + 1}. Time taken: {time.time() - start_time:.2f} seconds")

    close_output_files(out_f, q_f, a_f, m_f)


def evaluate(batch_id, prompter, prompts, answers, model, tokenizer,
             start_instance_id, model_name,
             out_f, q_f, a_f, m_f, 
             separator='++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'):
    batch_outputs = []

    for prompt_id, prompt in enumerate(prompts):
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(f'{batch_id} {prompt_id}')
        print(f'{prompt}')
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generation_output = model.generate(input_ids=input_ids, num_beams=1, num_return_sequences=1,
                                           max_new_tokens=256, temperature=0.15, top_p=0.95)
        output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        resp = prompter.get_response(output)
        print('Response:')
        print(resp)
        out_f.write(f'{separator}\n')
        out_f.write(f'{batch_id} {prompt_id}\n')
        out_f.write(f'{prompt}\n')
        if answers is not None:
            out_f.write(f'-----answer-----\n')
            out_f.write(f'{answers[prompt_id]}\n')
        out_f.write(f'-----{model_name}-----\n')
        out_f.write(f'{resp}\n')
        if q_f is not None:
            q = {'question_id': start_instance_id + prompt_id,
                 'turns': [prompt]}
            q_f.write(f"{json.dumps(q)}\n")
        if a_f is not None:
            a = {'question_id': start_instance_id + prompt_id,
                 'answer_id': f"oracle_{start_instance_id + prompt_id}",
                 'model_id': "oracle",
                 'choices': [{"index": 0, "turns": [answers[prompt_id]]}]}
            a_f.write(f"{json.dumps(a)}\n")
        m = {'question_id': start_instance_id + prompt_id,
             'answer_id': f"{model_name}_{start_instance_id + prompt_id}",
             'model_id': model_name,
             'choices': [{"index": 0, "turns": [resp]}]}
        m_f.write(f"{json.dumps(m)}\n")
        flush_output_files(out_f, q_f, a_f, m_f)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(main)
