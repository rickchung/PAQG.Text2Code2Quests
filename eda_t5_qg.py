"""
EDA: Fine-tune a T5 model for quesiton generation.
"""
from pathlib import Path

from datasets import load_from_disk
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer

from nn_data_processor import QGDataProcessor
from utils import freeze_embeds, get_my_logger

# Get a custom logger for this file
logger = get_my_logger(__name__)

# Arguments
base_model_name = 't5-small'
path_train_dataset = Path('Datasets', f'train_allquests_{base_model_name}')
path_valid_dataset = Path('Datasets', f'valid_allquests_{base_model_name}')
path_tuned_model = Path('Models', f'tuned_allquests_{base_model_name}')
target_quest_type = "what"

# Init a question type filter
if target_quest_type:
    type_list = target_quest_type.split(',')
    target_quest_filter = lambda item: item['quest_type'] in type_list
else:
    target_quest_filter = lambda x: True

# If the dataset has already existed, do not build a new one
if path_train_dataset.exists() or path_valid_dataset.exists():
    logger.warning("There are existing train/valid datasets. Load them from the disk.")
    # Load the existing train/valid datasets
    train_dataset = load_from_disk(path_train_dataset)
    valid_dataset = load_from_disk(path_valid_dataset)
else:
    logger.warning("Building a new dataset from scratch.")
    # Load a tokenizer
    tokenizer = T5Tokenizer.from_pretrained(base_model_name)
    # Load the dataset by the HF Datasets library
    data_processor = QGDataProcessor(tokenizer)
    data_processor.load_dataset()

    # Filter the dataset
    dataset = data_processor.dataset
    dataset = dataset.filter(target_quest_filter)

    # Extract train/valid dataset
    dataset_format = {'columns': ['input_ids', 'labels', 'attention_mask'], 'type': 'torch'}
    train_dataset = dataset['train']
    valid_dataset = dataset['validation']
    train_dataset.set_format(**dataset_format)
    valid_dataset.set_format(**dataset_format)
    # Save the train/valid datasets
    train_dataset.save_to_disk(path_train_dataset)
    valid_dataset.save_to_disk(path_valid_dataset)

    # Save the tokenizer
    data_processor.tokenizer.save_pretrained(path_tuned_model)

# Init a base model and fine-tune it
logger.info("Init the fine-tuning process")
model = T5ForConditionalGeneration.from_pretrained(base_model_name)
logger.info("Freeze the base model's embeddings.")
freeze_embeds(model)

training_args = TrainingArguments(
    num_train_epochs=1, output_dir='Results', logging_dir='Logs',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=1e-2,
)
trainer = Trainer(model=model, args=training_args,
                  train_dataset=train_dataset, eval_dataset=valid_dataset)
# Fine-tune the model
logger.info("Start fine-tuning")
trainer.train()
trainer.evaluate()

# Save the model
logger.info("Done fine-tuning")
model.save_pretrained(path_tuned_model)
