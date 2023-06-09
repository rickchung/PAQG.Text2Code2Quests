# !pip install transformers datasets pandas sentencepiece rouge_score spacy
# !pip install -U nltk

import subprocess
from pathlib import Path

from evaluate_generation import run_evaluate_translation, run_predict_valid_set
from fine_tune_t5qg import run_finetuning

# %%

train_shared_args = {
    'per_device_eval_batch_size': 32,
    'per_device_train_batch_size': 32,
    'base_model_name': 't5-small',
    'overwrite_output_dir': False,
    'reuse_existing_data': True,
    'dry': False,
    'question_labels': "what,when,why,where,when,how",
    'num_train_epochs': 1,
}

evaluate_shared_args = {
    'base_model_name': 't5-small',
    'ref_col': 'target_text',
    'question_labels': "what,when,why,where,when,how",
    'squad_test': True,
    'valid_size': 100,
    'batch_size': 64,
}

exp_args = [
    # Baseline 1
    {
        'tokenizer_args': "tc_tq,false,false",
    },
    # # Baseline 2
    # {
    #     'tokenizer_args': "tc_tqa,false,false",
    # },
    # # Exp 1
    # {
    #     'tokenizer_args': "tc_tq,true,false",
    # },
    # # Exp 2
    # {
    #     'tokenizer_args': "tc_tqa,true,false",
    # },
]

# storage_root = Path('/content/drive/MyDrive/Colab Notebooks/models/')
storage_root = Path('.')

# %%
# Fine tuning
# for kargs in exp_args:
#     model_path = run_finetuning(**{**kargs, **train_shared_args})
#     if model_path:
#         subprocess.run(['zip', '-r', Path(storage_root, f'{model_path}.zip'), model_path])

# %%
# Copy trained models from the storage
# !mkdir models
# !unzip '/content/drive/MyDrive/Colab Notebooks/models/tuned_what-when-why-where-when-how_tc-tq-false-false_t5-small.zip' -d '..'
# !unzip '/content/drive/MyDrive/Colab Notebooks/models/tuned_what-when-why-where-when-how_tc-tqa-false-false_t5-small.zip' -d '..'
# !unzip '/content/drive/MyDrive/Colab Notebooks/models/tuned_what-when-why-where-when-how_tc-tq-true-false_t5-small.zip' -d '..'
# !unzip '/content/drive/MyDrive/Colab Notebooks/models/tuned_what-when-why-where-when-how_tc-tqa-true-false_t5-small.zip' -d '..'


# %%
# Evaluation
for kargs in exp_args:
    path_predictions = run_predict_valid_set(**{**kargs, **evaluate_shared_args})
    store_cmd = ['zip', '-r', Path(storage_root, f'{path_predictions}.zip'), path_predictions]
    # subprocess.run(store_cmd)
    print(store_cmd)
    path_score = run_evaluate_translation(**{**kargs, **evaluate_shared_args})
    store_cmd = ['cp', path_score, Path(storage_root, path_predictions.parent, f'score_{path_predictions.name}.csv')]
    # subprocess.run(store_cmd)
    print(store_cmd)
