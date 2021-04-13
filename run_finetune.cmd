CALL conda activate depy3

rem TC -> Question + Answer
rem CALL python fine_tune_t5qg.py --question_labels "what,when,why,where,when,how" --tokenizer_args "tc_tqa,true,false" --num_train_epochs 5
rem CALL python evaluate_generation.py --squad --question_labels "what,when,why,where,when,how" --tokenizer_args "tc_tqa,true,false"

rem TC + A -> Question
CALL python fine_tune_t5qg.py --question_labels "what,when,why,where,when,how" --tokenizer_args "tca_tq,true,false" --num_train_epochs 3
CALL python evaluate_generation.py --squad --question_labels "what,when,why,where,when,how" --tokenizer_args "tca_tq,true,false"

rem TC -> Answers
rem CALL python fine_tune_t5qg.py --question_labels "what,when,why,where,when,how" --tokenizer_args "tc_ta,true,false" --num_train_epochs 5
rem CALL python evaluate_generation.py --squad --question_labels "what,when,why,where,when,how" --tokenizer_args "tc_ta,true,false"
