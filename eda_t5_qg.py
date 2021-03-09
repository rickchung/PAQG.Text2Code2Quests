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
target_quest_type = "what,how"
quest_type_label = target_quest_type.replace(',', '-')
base_model_name = 't5-small'

path_train_dataset = Path('Datasets', f'train_{quest_type_label}_{base_model_name}')
path_valid_dataset = Path('Datasets', f'valid_{quest_type_label}_{base_model_name}')
path_tuned_model = Path('Models', f'tuned_{quest_type_label}_{base_model_name}')


# Init a question type filter
def target_quest_filter(item):
    if target_quest_type:
        return item['quest_type'] in target_quest_type.split(',')
    return True


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

    # Get a tokenized dataset
    dataset = data_processor.tokenize_qg(quest_prefix=True)

    # Filter the dataset
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

# Peek the train dataset
logger.info(f'Train Dataset Summary: {train_dataset}')

# Init a base model and freeze the embeddings
logger.info("Init the fine-tuning process")
model = T5ForConditionalGeneration.from_pretrained(base_model_name)
logger.info("Freeze the base model's embeddings.")
freeze_embeds(model)

# Init a Trainer
training_args = TrainingArguments(
    output_dir='Results', logging_dir='Logs',
    num_train_epochs=3,
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
