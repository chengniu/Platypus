import json


def print_alpaca_data(alpaca_data_list, out_file, with_input_only, min_output_length):
    with open(out_file, 'w', encoding='UTF8') as f:
        for i, alpaca in enumerate(alpaca_data_list):
            if with_input_only and len(alpaca['input'].strip()) == 0:
                continue
            if min_output_length > 0 and len(alpaca['output'].strip().split(" ")) < min_output_length:
                continue
            f.write(f"{i}.+++++++++++++++++++++++++++++++++++++++++++++++++++\n")
            instruction = alpaca['instruction']
            f.write(f"instruction: {instruction}\n")
            input = alpaca['input']
            f.write(f"input: {input}\n")
            output = alpaca['output']
            f.write(f"output: {output}\n")



def print_alpaca_file(alpaca_file, out_file, with_input_only, min_output_length):
    lines = open(alpaca_file, 'r', encoding='UTF8').readlines()
    alpaca_data_list = [json.loads(x) for x in lines if len(x.strip()) > 0]
    return print_alpaca_data(alpaca_data_list, out_file, with_input_only, min_output_length)


if __name__ == "__main__":
    with_input_only = True
    min_output_length = 30
    print_alpaca_file('/home/dayong/chengniu/data/train-00000-of-00001-5b226e5ae97bf4b1.json', 
                      '/home/dayong/chengniu/data/train-00000-of-00001-5b226e5ae97bf4b1.txt', 
                      with_input_only=with_input_only, min_output_length=min_output_length)
    print_alpaca_file('/home/dayong/chengniu/data/bay_area_en_retrieved_gen.train.alpache.short.3.json', 
                      '/home/dayong/chengniu/data/bay_area_en_retrieved_gen.train.alpache.short.3.txt',
                      with_input_only=False, min_output_length=min_output_length)
