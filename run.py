from evaluate_generation import run_evaluate_generation
from fine_tune_t5qg import run_finetuning

train_shared_args = {
    'per_device_eval_batch_size': 64,
    'per_device_train_batch_size': 64,
    'overwrite_output_dir': True,
    'base_model_name': 't5-small',
    'reuse_existing_data': True,
    'dry': False,
}

evaluate_shared_args = {
    'base_model_name': 't5-small',
    'context_col': 'pre_context_cleaned',
    'squad_test': True,
    'dry': False,
}

exp_args = [
    {
        'question_labels': "what,when,why,where,when,how",
        'tokenizer_args': "tc_tq,true,false",
        'num_train_epochs': 2,
    },
    {
        'question_labels': "what,when,why,where,when,how",
        'tokenizer_args': "tc_tq,false,false",
        'num_train_epochs': 2,
    },
    {
        'question_labels': "what,when,why,where,when,how",
        'tokenizer_args': "tc_tqa,true,false",
        'num_train_epochs': 2,
    },
    {
        'question_labels': "what,when,why,where,when,how",
        'tokenizer_args': "tca_tq,true,false",
        'num_train_epochs': 2,
    },

]

for kargs in exp_args:
    run_finetuning(**{**kargs, **train_shared_args})
for kargs in exp_args:
    run_evaluate_generation(**{**kargs, **evaluate_shared_args})
