from pathlib import Path
from pprint import pprint

from transformers import T5Tokenizer, T5ForConditionalGeneration

# path_model = Path('Models', f'tuned_what_t5-small')
# path_model = Path('Models', f'tuned_what-how-which_t5-small')
path_model = Path('Models', f'tuned_what-how-which-where-who-other_t5-small')
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


test_texts = [
    'The code terminates the current line by writing the line separator string.',
    'You terminates the current line by writing the line separator string.',
]

for i in test_texts:
    pprint(ask_questions(i))
    print()
