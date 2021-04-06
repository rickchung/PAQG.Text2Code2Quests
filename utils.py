import logging

from torch import nn

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)


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
