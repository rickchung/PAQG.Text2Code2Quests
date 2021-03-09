from pathlib import Path

from transformers import T5Tokenizer, T5ForConditionalGeneration

path_model = Path('Models', f'tuned_allquests_t5-small')
model = T5ForConditionalGeneration.from_pretrained(path_model)
tokenizer = T5Tokenizer.from_pretrained(path_model)


def seq2seq(input_text):
    input_ids = tokenizer(input_text, return_tensors='pt')['input_ids']
    output_ids = model.generate(input_ids)
    output_text = tokenizer.decode(*output_ids)
    return output_text


test_texts = {
    'Java performs floating-point division when one or more operands are double values.',
    'Conveniently, the code for displaying a variable is the same regardless of its type.',
    'The following program converts a time of day to minutes.',
    'The value of minute is 59, and 59 divided by 60 should be 0.98333, not 0.',
}

for i in test_texts:
    print('context:', i, '\nquestion', seq2seq(i), '\n')
