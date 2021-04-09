import argparse
import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration


def ask_questions(context, question_labels, model, tokenizer):
    def seq2seq(input_text, quest_prefix=''):
        input_ids = tokenizer(quest_prefix + input_text, return_tensors='pt')['input_ids']
        output_ids = model.generate(input_ids)
        output_text = tokenizer.decode(*output_ids)
        return output_text

    return {k: seq2seq(context, f'k: ') for k in question_labels}


logging.basicConfig(
    format="%(asctime)s,%(levelname)s,%(name)s,%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filename='log_fine_tune_t5'
)

argparser = argparse.ArgumentParser()
argparser.add_argument('--base_model_name', default='t5-small')
argparser.add_argument('--tokenizer_args', required=True)
argparser.add_argument('--question_labels', required=True)
argparser.add_argument('--output', required=True)
argparser.add_argument('--dry', action='store_true')
args = argparser.parse_args()

# Base model
base_model_name = args.base_model_name
# Tokenizer arguments
tokenizer_args = args.tokenizer_args.split(',')
tokenizer_args_label = "_".join(tokenizer_args)
# Question types
target_quest_type = args.question_labels
quest_type_label = target_quest_type.replace(',', '-')
path_tuned_model = Path('models', f'tuned_{quest_type_label}_{tokenizer_args_label}_{base_model_name}')

# %%

logging.info('===== Generation process =====')

if not args.dry:
    # %% Load the textbook data for evaluation

    textbook = pd.read_json('data/thinkjava2.json')
    dataset = Dataset.from_pandas(textbook)
    dataset1 = dataset.filter(lambda ex: (ex['chapter'] == 'variables and operators'))

    # %% Build a simple QG pipeline

    question_labels = args.question_labels.split(',')
    model = T5ForConditionalGeneration.from_pretrained(path_tuned_model)
    tokenizer = T5Tokenizer.from_pretrained(path_tuned_model)

    input_dataset = dataset1
    context_col = 'pre_context_cleaned'
    out_columns = ['chapter', 'section', context_col] + question_labels

    logging.info("Generate targets")

    output_dataset = input_dataset.map(lambda ex: ask_questions(ex[context_col], question_labels, model, tokenizer))
    output_dataset.save_to_disk(str(Path("outputs", args.output)))
    output_dataset.to_csv(Path("outputs", f"{args.output}.csv"), columns=out_columns)
else:
    logging.info(f"Dry run. model={path_tuned_model} output={args.output}")
