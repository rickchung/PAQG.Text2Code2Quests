from pathlib import Path

from transformers import T5Tokenizer, T5ForConditionalGeneration

# path_model = Path('Models', f'tuned_what_t5-small')
path_model = Path('Models', f'tuned_what-how-which_t5-small')
# path_model = 't5-small'

model = T5ForConditionalGeneration.from_pretrained(path_model)
tokenizer = T5Tokenizer.from_pretrained(path_model)


def seq2seq(input_text, quest_prefix=''):
    input_ids = tokenizer(quest_prefix + '' + input_text, return_tensors='pt')['input_ids']
    output_ids = model.generate(input_ids)
    output_text = tokenizer.decode(*output_ids)
    return output_text


test_texts = [
    'Java has six relational operators that test the relationship between two values, for example, whether they are equal, or whether one is greater than the other.',
    'The result of a relational operator is one of two special values: true or false.',
    'To write useful programs, we almost always need to check conditions and react accordingly.',
    'The expression in parentheses is called the condition',
    'A for loop is definite, which means we know, at the beginning of the loop, how many times it will repeat.',
    'John bought some fruits.',
    # 'Diagrams like this one that show the state of the program are called memory diagrams.',
    # 'There\'s nothing wrong with a method like printTime, but it is not consistent with object-oriented style. A more idiomatic solution is to provide a special method called toString.'
]

for i in test_texts:
    print(f'Context: {i}\nWhat: {seq2seq(i, "what: ")}\nHow: {seq2seq(i, "how: ")}\nWhich: {seq2seq(i, "which: ")}\n')
