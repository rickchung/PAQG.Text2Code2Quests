import argparse
from pathlib import Path

import datasets
import pandas as pd
import spacy
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

import utils

logger = utils.get_my_logger(__name__)


def make_predictions(src_dataset: datasets.Dataset, path_model: Path,
                     src_col: str = 'source_text', target_qtypes: list = None,
                     add_src_prefix: bool = None, batch_size: int = 8):
    """
    Make predictions by `path_model` for `valid_data`[`source_col`]. The output dataset will always include the input
    columns. By default, the function uses the data format the same as the training set (in the column `source_text`).
    If `target_qtypes` is given, the output will include an additional column for each type of question. In this mode,
    you may decide whether to add the question type prefix in the source text.
    """
    # Load the trained model and tokenizer
    logger.info('Load the tuned model and the tokenizer')
    model = T5ForConditionalGeneration.from_pretrained(path_model)
    tokenizer = T5Tokenizer.from_pretrained(path_model)
    if torch.cuda.is_available():
        model.to('cuda')
    # For each item, generate all types of questions specified in `question_labels`
    logger.info('Make predictions')

    def post_process(text: str) -> str:
        return text.replace('<pad>', '').replace('</s>', '').strip()

    def seq2seq_batch(batch_examples):
        input_ids = tokenizer(batch_examples, padding=True, truncation=True, return_tensors='pt')['input_ids']
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')
        output_ids = model.generate(input_ids)
        output_text = tokenizer.batch_decode(output_ids)
        output_text = [post_process(i) for i in output_text]
        return output_text

    def seq2seq(src, src_prefix=''):
        """
        Make a prediction for `src` with `src_prefix` attached.
        """
        input_ids = tokenizer(src_prefix + src, return_tensors='pt')['input_ids']
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')
        output_ids = model.generate(input_ids)
        output_text = tokenizer.decode(*output_ids)
        # Remove special tokens
        output_text = post_process(output_text)
        return output_text

    # If a list of question types is given, make a prediction for each of them.
    if target_qtypes:
        # TODO: make this a batch function
        if add_src_prefix:
            rt = src_dataset.map(lambda e: {qt: seq2seq(e[src_col], f'{qt}: ') for qt in target_qtypes})
        else:
            rt = src_dataset.map(lambda e: {qt: seq2seq(e[src_col]) for qt in target_qtypes})
    # Otherwise, make one prediction and store the result in 'yhat'
    else:
        rt = src_dataset.map(lambda e: {'yhat': seq2seq_batch(e[src_col])}, batched=True, batch_size=batch_size)
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
    # target_qtypes = p_args['question_types']
    # tokenizer_args = p_args['tokenizer_args']
    # Output dir
    path_gen_questions = p_args['qg_output_path']
    # Source column
    # src_col = kargs['src_col']
    # The size of validation dataset to use
    valid_size = kargs.get('valid_size', -1)
    batch_size = kargs.get('batch_size', 8)

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
    rt = make_predictions(src_dataset, path_model, batch_size=batch_size)
    rt.save_to_disk(path_gen_questions)

    return path_gen_questions


def evaluate_translation(src_dataset: datasets.Dataset, ref_col: str, target_qtypes: list,
                         prediction_col: str = 'yhat', hl_token='highlight'):
    """
    Evaluate the given items by some offline metrics. The dataset must include the gold references (the target labels)
    and model predictions (in the columns `target_qtypes`). See the output of `make_predictions`.
    """

    # Load an NLP pipeline for text analysis
    # logger.info('Init SpaCy NLP pipeline')
    # nlp = spacy.load('en_core_web_sm')
    # nlp_pipe = lambda texts: list(nlp.pipe(texts, disable=["tagger", "parser", "ner", "lemmatizer", "textcat"]))

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

        logger.info('Process examples')

        def process_example(e):
            ref = e[ref_col]
            ref_tokens = ref.split()
            pred = e[prediction_col].split(hl_token)[0].strip()
            pred_tokens = pred.split()
            return {
                '_reference': ref, '_reference_tokens': ref_tokens,
                '_prediction': pred, '_prediction_tokens': pred_tokens
            }

        gen_items_qt = gen_items_qt.map(process_example)

        # Extract the gold references
        logger.info('Extract the gold references')
        references = gen_items_qt['_reference']
        references_tokens = gen_items_qt['_reference_tokens']
        # Extract the model predictions (only the question part)
        logger.info('Extract the model predictions')
        predictions = gen_items_qt['_prediction']
        predictions_tokens = gen_items_qt['_prediction_tokens']

        # Compute the score for the question type
        logger.info('Computer the scores...')
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
