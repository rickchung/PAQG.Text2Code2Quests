import argparse
from pathlib import Path

import datasets
import pandas as pd
import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration

import utils
from preprocess_textbook import get_qg_textbook

logger = utils.get_my_logger(__name__)


def generate_questions(context: str, question_labels: list, model, tokenizer):
    """
    A simple sequence-to-sequence QG pipeline
    """

    def seq2seq(input_text, quest_prefix=''):
        input_ids = tokenizer(quest_prefix + input_text, return_tensors='pt')['input_ids']
        output_ids = model.generate(input_ids)
        output_text = tokenizer.decode(*output_ids)
        # Remove special tokens
        output_text = output_text.replace('<pad>', '').replace('</s>', '').strip()
        return output_text

    return {k: seq2seq(context, f'{k}: ') for k in question_labels}


def evaluate_by_textbook(items, context_col, path_model, question_labels):
    """
    Generate questions from the selected textbook chapters.
    """
    model = T5ForConditionalGeneration.from_pretrained(path_model)
    tokenizer = T5Tokenizer.from_pretrained(path_model)
    rt = items.map(lambda _: generate_questions(_[context_col], question_labels, model, tokenizer))
    return rt


def evaluate_offline(items, source_col, target_col, path_model, question_labels, path_predictions):
    """
    Evaluate the given items by some offline metrics.
    """
    # Load an NLP pipeline for text analysis
    nlp = spacy.load('en_core_web_sm')
    nlp_pipe = lambda texts: list(nlp.pipe(
        texts, disable=["tagger", "parser", "ner", "lemmatizer", "textcat"]))
    # Load the trained model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(path_model)
    tokenizer = T5Tokenizer.from_pretrained(path_model)
    # For each item, generate all types of questions specified in `question_labels`
    gen_items = items.map(lambda _: generate_questions(_[source_col], question_labels, model, tokenizer))
    gen_items.save_to_disk(path_predictions)
    # Prepare different metrics
    metric_bleu = datasets.load_metric('bleu')
    metric_rouge = datasets.load_metric('rouge')
    metric_meteor = datasets.load_metric('meteor')
    performance = []
    for qt in question_labels:
        # Keep items that are associated with a certain type of questions
        items_qt = gen_items.filter(lambda _: _['quest_type'] == qt)
        if len(items_qt) == 0:
            logger.warning(f'No {qt} questions found. Skipped.')
            continue
        # Extract the gold references and the model predictions
        references = items_qt[target_col]
        references_tokens = [[j.text for j in i] for i in nlp_pipe(references)]
        predictions = items_qt[qt]
        predictions_tokens = [[j.text for j in i] for i in nlp_pipe(predictions)]
        # Compute the score for the question type
        bleu = metric_bleu.compute(predictions=predictions_tokens, references=[[i] for i in references_tokens])['bleu']
        rouge = metric_rouge.compute(predictions=predictions, references=references)
        rouge1_midf = rouge['rouge1'][1][2]
        rouge2_midf = rouge['rouge2'][1][2]
        rougel_midf = rouge['rougeL'][1][2]
        meteor = metric_meteor.compute(predictions=predictions, references=references)['meteor']
        performance.append({
            'qtype': qt, 'num_questions': len(items_qt),
            'bleu': bleu, 'meteor': meteor,
            'rouge1': rouge1_midf, 'rouge2': rouge2_midf, 'rougeL': rougel_midf,
        })
    performance = pd.DataFrame(performance)

    return gen_items, performance


def run_evaluate_generation(**kargs):
    p_args = utils.process_args(kargs['base_model_name'], kargs['tokenizer_args'], kargs['question_labels'])
    path_model = p_args['path_tuned_model']
    path_tokenized_data = p_args['path_tokenized_dataset']
    path_gen_questions = p_args['qg_output_path']
    quest_types = p_args['question_types']

    dry = kargs['dry']
    squad_teset = kargs['squad_test']
    context_col = kargs['context_col']

    # %%

    logger.info('===== Generation process =====')

    if not dry:
        # Evaluate by the SQuAD dataset
        if squad_teset and path_tokenized_data.exists():
            # Load the validation set
            dataset = datasets.load_from_disk(str(path_tokenized_data))['validation']
            logger.info("Evaluate by the SQuAD test set")
            out, scores = evaluate_offline(dataset, 'source_text', 'target_text', path_model, quest_types,
                                           path_gen_questions)
            out.to_csv(f"{path_gen_questions}_squad.csv", columns=['source_text', 'target_text'] + quest_types)
            scores.to_csv(Path(path_gen_questions, f'offline_scores.csv'))

        else:
            logger.info("Evaluate by the textbook dataset")
            dataset = get_qg_textbook(['variables and operators'])
            out = evaluate_by_textbook(dataset, context_col, path_model, quest_types)
            out_columns = ['chapter', 'section', context_col] + quest_types
            out.save_to_disk(path_gen_questions)
            out.to_csv(f"{path_gen_questions}.csv", columns=out_columns)
    else:
        logger.info(f"Dry run. model={path_model} output={path_gen_questions}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--base_model_name', default='t5-small')
    argparser.add_argument('--tokenizer_args', required=True)
    argparser.add_argument('--question_labels', required=True)
    argparser.add_argument('--context_col', default='pre_context_cleaned')
    argparser.add_argument('--squad_test', action='store_true')
    argparser.add_argument('--dry', action='store_true')
    args = argparser.parse_args()
    kargs = args.__dict__

    run_evaluate_generation(**kargs)
