import logging
from pathlib import Path

from torch import nn


def process_args(base_model_name: str, tokenizer_args: str, question_labels: str):
    """
    Shared model/data/metadata information.
    """
    qt_id = question_labels.replace(',', '-')
    question_types = question_labels.split(',')
    tk_args_id = tokenizer_args.replace(',', '-')
    tokenizer_args_list = tokenizer_args.split(',')

    path_tuned_model = Path('models', f'tuned_{qt_id}_{tk_args_id}_{base_model_name}')
    path_tokenized_dataset = Path('models', 'tokenized_data', f'{tk_args_id}_{base_model_name}')
    path_train_dataset = Path(path_tuned_model, f'train_{tk_args_id}_{base_model_name}_{qt_id}')
    path_valid_dataset = Path(path_tuned_model, f'valid_{tk_args_id}_{base_model_name}_{qt_id}')
    qg_output_path = Path("outputs", f'{tk_args_id}_{qt_id}_{base_model_name}')
    return {
        'qt_id': qt_id,
        'question_types': question_types,
        'tk_args_id': tk_args_id,
        'tokenizer_args': tokenizer_args_list,

        'base_model_name': base_model_name,

        'path_tuned_model': path_tuned_model,
        'path_tokenized_dataset': path_tokenized_dataset,
        'path_train_dataset': path_train_dataset,
        'path_valid_dataset': path_valid_dataset,

        'qg_output_path': qg_output_path
    }


def get_my_logger(logger_name):
    """
    Create a custom logger with the given name `logger_name`.
    """
    logger = logging.getLogger(logger_name)
    return logger


def freeze_params(model: nn.Module):
    for i in model.parameters(): i.requires_grad = False


def freeze_embeds(model: nn.Module):
    """
    Freeze the embeddings of T5
    """
    freeze_params(model.shared)
    freeze_params(model.encoder.embed_tokens)
    freeze_params(model.decoder.embed_tokens)
