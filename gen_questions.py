from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

# %% ==================== Load the textbook data for evaluation ====================

textbook = pd.read_json('data/thinkjava2.json')
dataset = Dataset.from_pandas(textbook)

# Only one chapter, plain context paragraphs without code
dataset1 = dataset.filter(lambda ex: (ex['chapter'] == 'variables and operators') and (ex['code'] == ''))

# %% ==================== Build a simple QG pipeline ====================

path_model = Path('Models', f'tuned_what-how-which-where-who-other_t5-small')
# path_model = Path('Models', f'tuned_what_t5-small')
# path_model = Path('Models', f'tuned_what-how-which_t5-small')
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
        'context': context,
        'what': seq2seq(context, "what: "),
        'how': seq2seq(context, "how: "),
        'which': seq2seq(context, "which: "),
        'where': seq2seq(context, "where: "),
        'who': seq2seq(context, "who: "),
        'other': seq2seq(context, "other: "),
    }


input_dataset = dataset1
output_dataset = input_dataset.map(lambda ex: ask_questions(ex['pre_context']))
output_dataset.save_to_disk(Path("outputs", "gen_questions_sample"))
