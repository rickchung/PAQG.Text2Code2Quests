"""
EDA: Fine-tune a T5 model for question generation.
"""
import argparse
import logging
from pathlib import Path

import datasets
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer

from preprocess_train_data import QGDataProcessor
from utils import freeze_embeds

logging.basicConfig(
    format="%(asctime)s,%(levelname)s,%(name)s,%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filename='log_fine_tune_t5'
)

# Arguments
argparser = argparse.ArgumentParser()
# trainer arguments
argparser.add_argument('--num_train_epochs', default=1, type=int)
argparser.add_argument('--per_device_eval_batch_size', default=8, type=int)
argparser.add_argument('--per_device_train_batch_size', default=8, type=int)
argparser.add_argument('--overwrite_output_dir', default="true")
argparser.add_argument('--question_labels', default="what,how,which,where,who,why,other")
argparser.add_argument('--base_model_name', default='t5-small')
argparser.add_argument('--reuse_existing_data', default="true")
argparser.add_argument('--dry', action='store_true')
# dataset arguments
argparser.add_argument('--tokenizer_args', default='tc_tq,true,true')
args = argparser.parse_args()

# %%

# Tokenizer arguments
tokenizer_args = args.tokenizer_args.split(',')
tokenizer_args_label = "_".join(tokenizer_args)
# Overwrite the output
args.overwrite_output_dir = (args.overwrite_output_dir.lower() == "true")
# Reuse the dataset
args.reuse_existing_data = (args.reuse_existing_data.lower() == "true")
# Question type filters
target_quest_type = args.question_labels
quest_type_label = target_quest_type.replace(',', '-')
# Base model name
base_model_name = args.base_model_name
# Output paths
path_tokenized_dataset = Path('models', 'tokenized_data', f'{tokenizer_args_label}_{base_model_name}')
path_tuned_model = Path('models', f'tuned_{quest_type_label}_{tokenizer_args_label}_{base_model_name}')
path_train_dataset = Path(path_tuned_model, f'train_{tokenizer_args_label}_{base_model_name}_{quest_type_label}')
path_valid_dataset = Path(path_tuned_model, f'valid_{tokenizer_args_label}_{base_model_name}_{quest_type_label}')

# %%

logging.info('===== Fine-tuning process =====')
logging.info('Arguments: ' + str(args))

# If the tokenzied dataset has already existed, do not build a new one
if args.reuse_existing_data and path_tokenized_dataset.exists():
    logging.warning(f"Reuse the existing data: {path_tokenized_dataset}")
    dataset = datasets.load_from_disk(str(path_tokenized_dataset))
else:
    logging.warning("Building a new dataset from scratch")
    # Load a tokenizer
    tokenizer = T5Tokenizer.from_pretrained(base_model_name)
    # Load the dataset by the HF Datasets library
    data_processor = QGDataProcessor(tokenizer)
    data_processor.load_dataset()
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
        logging.error(f"Unknown tokenizer: {tokenizer_args[0]}")
        raise Exception(f"Unknown tokenizer: {tokenizer_args[0]}")
    # Save the tokenized dataset (in the data folder) and the tokenizer (in the tuned model folder)
    dataset.save_to_disk(path_tokenized_dataset)
    data_processor.tokenizer.save_pretrained(path_tuned_model)

# %%

# Apply the question type filter
if target_quest_type:
    logging.info("Apply the question type filter")
    quest_type_labels = target_quest_type.split(',')
    dataset = dataset.filter(lambda x: x['quest_type'] in quest_type_labels)
    for i in quest_type_labels:
        count = len(dataset['train'].filter(lambda x: x['quest_type'] == i))
        logging.info(f'{i} question counts = {count}')

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

# Init a base model and freeze the embeddings
logging.info("Init the fine-tuning process")
model = T5ForConditionalGeneration.from_pretrained(base_model_name)
logging.info("Freeze the base model's embeddings.")
freeze_embeds(model)

# Init a Trainer
training_args = TrainingArguments(
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    warmup_steps=500,
    weight_decay=1e-2,
    save_steps=5000,
    output_dir=str(path_tuned_model),
    overwrite_output_dir=args.overwrite_output_dir

)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=valid_dataset)

# Fine-tune the model
if not args.dry:
    logging.info(f"Start fine-tuning (resume: {not args.overwrite_output_dir})")
    trainer.train()
    trainer.evaluate()
    # Save the model
    logging.info("Done")
    model.save_pretrained(path_tuned_model)
else:
    logging.info(f'Dry. Do nothing (resume: {not args.overwrite_output_dir})')

