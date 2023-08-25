from experiments.data_preparation.data_util import merge_shuffle_jsonl


def yelp_qa_to_alpaca(in_file, out_file, mt_probabilies):
    with open(in_file, 'r', encoding='UTF8') as in_f:
        with open(out_file, 'w', encoding='UTF8') as out_f:
