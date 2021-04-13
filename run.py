import subprocess

from evaluate_generation import run_evaluate_generation
from fine_tune_t5qg import run_finetuning

train_shared_args = {
    'per_device_eval_batch_size': 32,
    'per_device_train_batch_size': 32,
    'base_model_name': 't5-small',
    'overwrite_output_dir': True,
    'reuse_existing_data': True,
    'dry': False,
    'question_labels': "what,when,why,where,when,how",
    'num_train_epochs': 1,
}

evaluate_shared_args = {
    'base_model_name': 't5-small',
    'context_col': 'pre_context_cleaned',
    'squad_test': True,
    'dry': False,
}

exp_args = [
    # Baseline 1
    {
        'tokenizer_args': "tc_tq,false,false",
    },
    # Baseline 2
    {
        'tokenizer_args': "tc_tqa,false,false",
    },
    # Exp 1
    {
        'tokenizer_args': "tc_tq,true,false",
    },
    # Exp 2
    {
        'tokenizer_args': "tc_tqa,true,false",
    },
]

for kargs in exp_args:
    model_path = run_finetuning(**{**kargs, **train_shared_args})
    subprocess.run(['zip', '-r', f'{model_path}.zip', model_path])

for kargs in exp_args:
    run_evaluate_generation(**{**kargs, **evaluate_shared_args})
