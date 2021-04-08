import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', required=True)
argparser.add_argument('--output', required=True)
args = argparser.parse_args()

# %% ==================== Load the textbook data for evaluation ====================

textbook = pd.read_json('data/thinkjava2.json')
dataset = Dataset.from_pandas(textbook)
# Only one chapter, plain context paragraphs without code
dataset1 = dataset.filter(lambda ex: (ex['chapter'] == 'variables and operators') and (ex['code'] == ''))

# %% ==================== Build a simple QG pipeline ====================

path_model = Path('models', args.model)
# path_model = 't5-small'

model = T5ForConditionalGeneration.from_pretrained(path_model)
tokenizer = T5Tokenizer.from_pretrained(path_model)


def ask_questions(context):
    def seq2seq(input_text, quest_prefix=''):
        input_ids = tokenizer(quest_prefix + '' + input_text, return_tensors='pt')['input_ids']
        output_ids = model.generate(input_ids)
        output_text = tokenizer.decode(*output_ids)
        return output_text

    return {
        'what': seq2seq(context, "what: "),
        'who': seq2seq(context, "who: "),
        'which': seq2seq(context, "which: "),
        'where': seq2seq(context, "where: "),
        'why': seq2seq(context, "why: "),
        'how': seq2seq(context, "how: "),
        'other': seq2seq(context, "other: "),
    }


input_dataset = dataset1
output_dataset = input_dataset.map(lambda ex: ask_questions(ex['hl_pre_context']))
output_dataset.save_to_disk(Path("outputs", args.output))
output_dataset.to_csv(
    Path("outputs", f"{args.output}.csv"),
    columns=['chapter', 'section', 'hl_pre_context', 'who', 'where', 'what', 'which', 'why', 'how', 'other'])
