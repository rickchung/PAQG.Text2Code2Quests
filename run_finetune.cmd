CALL conda activate paqg2

@REM TC -> Question + Answer
@REM CALL python fine_tune_t5qg.py --question_labels "what,when,why,where,when,how" --tokenizer_args "tc_tqa,true,false" --num_train_epochs 5
@REM CALL python evaluate_generation.py --squad --question_labels "what,when,why,where,when,how" --tokenizer_args "tc_tqa,true,false"

@REM TC + A -> Question
@REM CALL python fine_tune_t5qg.py --question_labels "what,when,why,where,when,how" --tokenizer_args "tca_tq,true,false" --num_train_epochs 3
CALL python evaluate_generation.py --squad --question_labels "what,when,why,where,when,how" --tokenizer_args "tca_tq,true,false"

@REM TC -> Answers
@REM CALL python fine_tune_t5qg.py --question_labels "what,when,why,where,when,how" --tokenizer_args "tc_ta,true,false" --num_train_epochs 5
@REM CALL python evaluate_generation.py --squad --question_labels "what,when,why,where,when,how" --tokenizer_args "tc_ta,true,false"

@REM CALL python fine_tune_t5qg.py --question_labels "what,how" --tokenizer_args "tc_tq,true,false" --num_train_epochs 3
@REM CALL python evaluate_generation.py --squad --question_labels "what,how" --tokenizer_args "tc_tq,true,false"
