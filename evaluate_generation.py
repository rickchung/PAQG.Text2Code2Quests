import argparse
import logging

import datasets
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration

import utils
from preprocess_textbook import get_qg_textbook


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


def evaluate_by_textbook(items, context_col, path_model, question_labels):
    """
    Generate questions from the selected textbook chapters.
    """
    model = T5ForConditionalGeneration.from_pretrained(path_model)
    tokenizer = T5Tokenizer.from_pretrained(path_model)
    output_dataset = items.map(lambda ex: ask_questions(ex[context_col], question_labels, model, tokenizer))
    return output_dataset


def evaluate_offline(items, source_col, target_col, question_labels, path_model):
    """
    Evaluate the given items by some offline metrics.
    """
    model = T5ForConditionalGeneration.from_pretrained(path_model)
    tokenizer = T5Tokenizer.from_pretrained(path_model)
    pass


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
    argparser.add_argument('--squad_test', action='store_true')
    argparser.add_argument('--sample', default=-1, type=int)
    argparser.add_argument('--dry', action='store_true')
    args = argparser.parse_args()

    p_args = utils.process_args(args.base_model_name, args.tokenizer_args, args.question_labels)

    # %%

    logging.info('===== Generation process =====')

    if not args.dry:
        if args.squad_test and p_args['path_tokenized_dataset'].exists():
            dataset = datasets.load_from_disk(str(p_args['path_tokenized_dataset']))['validation']
            if args.sample:
                x = np.random.randint([len(dataset)] * args.sample)
                dataset = dataset.select(x)
            logging.info("Evaluate by the SQuAD test set")
            # evaluate_by_textbook(dataset, args.context_col, p_args['path_tuned_model'], p_args['question_types'])
        else:
            logging.info("Evaluate by the textbook dataset")
            dataset = get_qg_textbook(['variables and operators'])
            out = evaluate_by_textbook(dataset, args.context_col, p_args['path_tuned_model'], p_args['question_types'])
            out_columns = ['chapter', 'section', args.context_col] + p_args['question_types']
            out.save_to_disk(p_args['qg_output_path'])
            out.to_csv(f"{p_args['qg_output_path']}.csv", columns=out_columns)
    else:
        logging.info(f"Dry run. model={p_args['path_tuned_model']} output={p_args['qg_output_path']}")
