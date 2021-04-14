import argparse
from pathlib import Path

import datasets
import pandas as pd
import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration

import utils

logger = utils.get_my_logger(__name__)


def make_predictions(src_dataset: datasets.Dataset, src_col: str, path_model: Path, target_qtypes: list,
                     add_src_prefix: bool):
    """
    Make predictions by `path_model` for `valid_data`[`source_col`]. The output dataset will include the input columns
    and an additional column for each type of question.
    """
    # Load the trained model and tokenizer
    logger.info('Load the tuned model and the tokenizer')
    model = T5ForConditionalGeneration.from_pretrained(path_model)
    tokenizer = T5Tokenizer.from_pretrained(path_model)
    # For each item, generate all types of questions specified in `question_labels`
    logger.info('Make predictions')

    def seq2seq(src, src_prefix=''):
        """
        Make a prediction for `src` with `src_prefix` attached.
        """
        input_ids = tokenizer(src_prefix + src, return_tensors='pt')['input_ids']
        output_ids = model.generate(input_ids)
        output_text = tokenizer.decode(*output_ids)
        # Remove special tokens
        output_text = output_text.replace('<pad>', '').replace('</s>', '').strip()
        return output_text

    if add_src_prefix:
        rt = src_dataset.map(lambda e: {qt: seq2seq(e[src_col], f'{qt}: ') for qt in target_qtypes})
    else:
        rt = src_dataset.map(lambda e: {k: seq2seq(e[src_col]) for k in target_qtypes})

    return rt


def run_predict_valid_set(**kargs):
    """
    A wrapper of the `make_prediction` method. This method only make predictions for the validation dataset in the
    model folder.

    :return: the path to the saved prediction data.
    """
    logger.info(f'Run predict valid set {kargs}')
    # Parameters following the tuned model and tokenizer
    p_args = utils.process_args(kargs['base_model_name'], kargs['tokenizer_args'], kargs['question_labels'])
    path_model = p_args['path_tuned_model']
    path_valid_dataset = p_args['path_valid_dataset']
    target_qtypes = p_args['question_types']
    tokenizer_args = p_args['tokenizer_args']
    # Output dir
    path_gen_questions = p_args['qg_output_path']
    # Source column
    src_col = kargs['src_col']
    # The size of validation dataset to use
    valid_size = kargs.get('valid_size', -1)

    # The validation dataset must exist in the model folder
    if not path_valid_dataset.exists():
        raise FileNotFoundError(f'validation dataset does not exist: {path_valid_dataset}')

    # Load the validation set
    src_dataset = datasets.load_from_disk(str(path_valid_dataset))
    src_dataset.reset_format()
    # If the sample size N is given, get the first N rows
    if valid_size > 0:
        logger.info(f'Limit the sample size to the first {valid_size} items')
        src_dataset = src_dataset.select([i for i in range(valid_size)])

    # Make predictions
    # `tokenizer_args[1]` indicates if the source text needs a question type prefix.
    # This setting follows the fine-tuning schema of the tuned model.
    rt = make_predictions(src_dataset, src_col, path_model, target_qtypes, tokenizer_args[1])
    rt.save_to_disk(path_gen_questions)

    return path_gen_questions


def evaluate_translation(src_dataset: datasets.Dataset, ref_col: str, target_qtypes: list,
                         hl_token='highlight'):
    """
    Evaluate the given items by some offline metrics. The dataset must include the gold references (the target labels)
    and model predictions (in the columns `target_qtypes`). See the output of `make_predictions`.
    """

    # Load an NLP pipeline for text analysis
    logger.info('Init SpaCy NLP pipeline')
    nlp = spacy.load('en_core_web_sm')
    nlp_pipe = lambda texts: list(nlp.pipe(texts, disable=["tagger", "parser", "ner", "lemmatizer", "textcat"]))

    # Prepare different metrics
    logger.info('Prepare metrics')
    metric_bleu = datasets.load_metric('bleu')
    metric_rouge = datasets.load_metric('rouge')
    metric_meteor = datasets.load_metric('meteor')

    # Evaluate the translation performance for each type of questions
    performance = []
    for qt in target_qtypes:
        logger.info(f"Evaluate {qt} questions")
        # Keep items that are associated with a certain type of questions
        gen_items_qt = src_dataset.filter(lambda _: _['quest_type'] == qt)
        if len(gen_items_qt) == 0:
            logger.warning(f'No {qt} questions found. Skipped.')
            continue
        # Extract the gold references
        references = gen_items_qt[ref_col]
        references_tokens = [[j.text for j in i] for i in nlp_pipe(references)]
        # Extract the model predictions (only the question part)
        predictions = gen_items_qt.map(lambda e: {'quest': e[qt].split(hl_token)[0].strip()})['quest']
        predictions_tokens = [[j.text for j in i] for i in nlp_pipe(predictions)]

        # Compute the score for the question type

        references_bleu = [[i] for i in references_tokens]
        bleu1 = metric_bleu.compute(predictions=predictions_tokens, references=references_bleu, max_order=1)['bleu']
        bleu2 = metric_bleu.compute(predictions=predictions_tokens, references=references_bleu, max_order=2)['bleu']
        bleu3 = metric_bleu.compute(predictions=predictions_tokens, references=references_bleu, max_order=3)['bleu']
        bleu4 = metric_bleu.compute(predictions=predictions_tokens, references=references_bleu, max_order=4)['bleu']

        rouge = metric_rouge.compute(predictions=predictions, references=references)
        rouge1_midf = rouge['rouge1'][1][2]
        rouge2_midf = rouge['rouge2'][1][2]
        rougel_midf = rouge['rougeL'][1][2]

        meteor = metric_meteor.compute(predictions=predictions, references=references)['meteor']

        performance.append({
            'qtype': qt, 'num_questions': len(gen_items_qt),
            'bleu1': bleu1, 'bleu2': bleu2, 'bleu3': bleu3, 'bleu4': bleu4,
            'meteor': meteor,
            'rouge1': rouge1_midf, 'rouge2': rouge2_midf, 'rougeL': rougel_midf,
        })

    performance = pd.DataFrame(performance)

    return performance


def run_evaluate_translation(**kargs):
    """
    A wrapper for the method `evaluate_translation`.

    :return: the path to the saved score.
    """
    logger.info(f'Run evaluate translation {kargs}')
    p_args = utils.process_args(kargs['base_model_name'], kargs['tokenizer_args'], kargs['question_labels'])
    path_gen_questions = p_args['qg_output_path']
    quest_types = p_args['question_types']
    ref_col = kargs['ref_col']
    test_on_squad = kargs.get('squad_test', False)
    squad_test_size = kargs.get('squad_test_size', -1)

    # Evaluate by the SQuAD dataset
    path_scores = None
    if test_on_squad:
        logger.info("Evaluate by the SQuAD test set")

        # The validation dataset (including predictions) must exist in the model folder
        if not path_gen_questions.exists():
            raise FileNotFoundError(f'validation dataset does not exist: {path_gen_questions}')
        # Load the validation set
        valid_set = datasets.load_from_disk(str(path_gen_questions))
        valid_set.reset_format()
        # If the sample size N is given, get the first N rows
        if squad_test_size > 0:
            logger.info(f'Limit the sample size to the first {squad_test_size} items')
            valid_set = valid_set.select([i for i in range(squad_test_size)])

        # Compute the offline score
        scores = evaluate_translation(valid_set, ref_col, quest_types)
        # Save the result
        path_scores = Path(path_gen_questions, f'offline_scores.csv')
        scores.to_csv(path_scores)
    logger.info('Done')

    return path_scores


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

    run_evaluate_translation(**kargs)
