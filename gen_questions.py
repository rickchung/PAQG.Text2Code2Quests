import argparse
import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

import utils


def get_qg_textbook(chapters=()):
    """
    Load the default textbook content.
    """
    textbook = pd.read_json('data/thinkjava2.json')
    dataset = Dataset.from_pandas(textbook)

    if chapters:
        dataset = dataset.filter(lambda ex: (ex['chapter'] in chapters))

    return dataset


def ask_questions(context: str, question_labels: list, model, tokenizer):
    """
    A simple sequence-to-sequence QG pipeline
    """

    def seq2seq(input_text, quest_prefix=''):
        input_ids = tokenizer(quest_prefix + input_text, return_tensors='pt')['input_ids']
        output_ids = model.generate(input_ids)
        output_text = tokenizer.decode(*output_ids)
        return output_text

    return {k: seq2seq(context, f'{k}: ') for k in question_labels}


def evaluate_qg_textbook(textbook_dataset, context_col, path_model, question_labels: list, output_path):
    """
    Generate questions from the selected textbook chapters.
    """
    model = T5ForConditionalGeneration.from_pretrained(path_model)
    tokenizer = T5Tokenizer.from_pretrained(path_model)

    output_dataset = textbook_dataset.map(lambda ex: ask_questions(ex[context_col], question_labels, model, tokenizer))
    out_columns = ['chapter', 'section', context_col] + question_labels

    output_dataset.save_to_disk(output_path)
    output_dataset.to_csv(f"{output_path}.csv", columns=out_columns)

    return output_dataset


if __name__ == '__main__':
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
    argparser.add_argument('--context_col', default='pre_context_cleaned')
    argparser.add_argument('--dry', action='store_true')
    args = argparser.parse_args()

    p_args = utils.process_args(args.base_model_name, args.tokenizer_args, args.question_labels)

    # %%

    logging.info('===== Generation process =====')

    if not args.dry:
        dataset = get_qg_textbook(['variables and operators'])
        evaluate_qg_textbook(
            dataset, args.context_col,
            p_args['path_tuned_model'],
            p_args['question_types'],
            p_args['qg_output_path'])
    else:
        logging.info(f"Dry run. model={p_args['path_tuned_model']} output={p_args['qg_output_path']}")
