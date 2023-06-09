"""
EDA: Fine-tune a T5 model for question generation. References:
- https://huggingface.co/t5-small
- https://huggingface.co/transformers/custom_datasets.html
"""
import argparse

import datasets
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer

import utils
from preprocess_train_data import QgDataProcessor

logger = utils.get_my_logger(__name__)


def run_finetuning(**kargs):
    # Processed arguments
    p_args = utils.process_args(kargs['base_model_name'], kargs['tokenizer_args'], kargs['question_labels'])
    base_model_name = p_args['base_model_name']
    path_tokenized_dataset = p_args['path_tokenized_dataset']
    tokenizer_args = p_args['tokenizer_args']
    path_tuned_model = p_args['path_tuned_model']
    path_train_dataset = p_args['path_train_dataset']
    path_valid_dataset = p_args['path_valid_dataset']
    question_types = p_args['question_types']
    # Other key arguments
    reuse_existing_data = kargs['reuse_existing_data']
    num_train_epochs = kargs['num_train_epochs']
    per_device_train_batch_size = kargs['per_device_train_batch_size']
    per_device_eval_batch_size = kargs['per_device_eval_batch_size']
    overwrite_output_dir = kargs['overwrite_output_dir']
    dry = kargs['dry']

    # %%

    logger.info('===== Fine-tuning process =====')
    logger.info('Arguments: ' + str(kargs))

    # Load a tokenizer
    tokenizer = T5Tokenizer.from_pretrained(base_model_name)
    # Load the base dataset
    data_processor = QgDataProcessor(tokenizer)
    data_processor.load_dataset()

    # If the tokenzied dataset has already existed, do not build a new one
    if reuse_existing_data and path_tokenized_dataset.exists():
        logger.warning(f"Reuse the existing data: {path_tokenized_dataset}")
        dataset = datasets.load_from_disk(str(path_tokenized_dataset))
    else:
        logger.warning("Building a new dataset from scratch")
        # Get the tokenized dataset
        if tokenizer_args[0] == 'tc_tq':
            dataset = data_processor.get_tokenized_tc_tq(*tokenizer_args[1:])
        elif tokenizer_args[0] == 'tc_ta':
            dataset = data_processor.get_tokenized_tc_ta(*tokenizer_args[1:])
        elif tokenizer_args[0] == 'tc_tqa':
            dataset = data_processor.get_tokenized_tc_tqa(*tokenizer_args[1:])
        elif tokenizer_args[0] == 'tca_tq':
            dataset = data_processor.get_tokenized_tca_tq(*tokenizer_args[1:])
        else:
            logger.error(f"Unknown tokenizer: {tokenizer_args[0]}")
            raise Exception(f"Unknown tokenizer: {tokenizer_args[0]}")
        # Save the tokenized dataset (in the data folder) and the tokenizer (in the tuned model folder)
        dataset.save_to_disk(path_tokenized_dataset)

    # %%

    # Apply the question type filter
    if question_types:
        logger.info("Apply the question type filter")
        dataset = dataset.filter(lambda x: x['quest_type'] in question_types)

    # Extract train/valid dataset
    dataset_format = {'columns': ['input_ids', 'labels', 'attention_mask'], 'type': 'torch'}
    train_dataset = dataset['train']
    valid_dataset = dataset['validation']
    train_dataset.set_format(**dataset_format)
    valid_dataset.set_format(**dataset_format)
    # Save the train/valid datasets
    train_dataset.save_to_disk(path_train_dataset)
    valid_dataset.save_to_disk(path_valid_dataset)

    # %%

    # Check if a tuned model has already existed
    if not overwrite_output_dir and path_tuned_model.exists():
        logger.warn(f'A tuned model has already existed. Stop: {path_tuned_model}')
        return None

    # Init a base model and freeze the embeddings
    logger.info("Init the fine-tuning process")
    model = T5ForConditionalGeneration.from_pretrained(base_model_name)
    logger.info("Freeze the base model's embeddings.")
    utils.freeze_embeds(model)

    # Init a Trainer
    training_args = TrainingArguments(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=500,
        weight_decay=1e-5,
        save_steps=5000,
        output_dir=str(path_tuned_model),
        overwrite_output_dir=overwrite_output_dir,

    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=valid_dataset)

    # Fine-tune the model
    if not dry:
        logger.info(f"Start fine-tuning (resume: {not overwrite_output_dir})")
        trainer.train()
        # Save the model
        model.save_pretrained(path_tuned_model)
        # Save the tokenizer in the model folder
        data_processor.tokenizer.save_pretrained(path_tuned_model)
        logger.info("Done")
    else:
        logger.info(f'Dry. Do nothing (resume: {not overwrite_output_dir})')

    return path_tuned_model


if __name__ == '__main__':
    # Arguments
    argparser = argparse.ArgumentParser()
    # trainer arguments
    argparser.add_argument('--num_train_epochs', default=1, type=int)
    argparser.add_argument('--per_device_eval_batch_size', default=8, type=int)
    argparser.add_argument('--per_device_train_batch_size', default=8, type=int)
    argparser.add_argument('--overwrite_output_dir', default="true")
    argparser.add_argument('--question_labels', default="what,how,which,where,who,why,other")
    argparser.add_argument('--base_model_name', default='t5-small')
    argparser.add_argument('--reuse_existing_data', default=True)
    argparser.add_argument('--dry', action='store_true')
    # dataset arguments
    argparser.add_argument('--tokenizer_args', default='tc_tq,true,true')

    args = argparser.parse_args()
    kargs = args.__dict__
    run_finetuning(**kargs)
