from random import shuffle
import json



def merge_shuffle_jsonl(in_files, out_file):
    lines = []
    for in_file in in_files:
        with open(in_file, encoding='UTF8') as f:
            lines_1 = f.readlines()
            [lines.append(x.strip()) for x in lines_1 if len(x.strip()) > 0]
    shuffle(lines)
    [json.loads(x) for x in lines]
    with open(out_file, 'w', encoding='UTF8') as f:
        for line in lines:
            f.write(f"{line}\n")

