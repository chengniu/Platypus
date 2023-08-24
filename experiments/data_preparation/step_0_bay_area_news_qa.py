from experiments.data_preparation.data_util import merge_shuffle_jsonl


if __name__ == "__main__":
    input_files = ['/home/dayong/chengniu/data/train-00000-of-00001-5b226e5ae97bf4b1.json',
                   '/home/dayong/chengniu/data/bay_area_en_retrieved_gen.train.alpache.short.3.json']
    out_file = '/home/dayong/chengniu/data/platypus_bay_area_news.json'
    merge_shuffle_jsonl(input_files, out_file)

    