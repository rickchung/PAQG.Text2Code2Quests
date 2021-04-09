CALL conda activate depy3

rem CALL python fine_tune_t5qg.py --question_labels "what,when,why,how" --tokenizer_args "tc_tq,true,false"
rem CALL python fine_tune_t5qg.py --question_labels "what,when,why,how" --tokenizer_args "tc_ta,true,false"
rem CALL python fine_tune_t5qg.py --question_labels "what,when,why,how" --tokenizer_args "tc_tqa,true,false"
rem CALL python fine_tune_t5qg.py --question_labels "what,when,why,how" --tokenizer_args "tca_tq,true,false"

rem CALL python gen_questions.py --question_labels "what,when,why,how" --tokenizer_args "tc_tq,true,false"
rem CALL python gen_questions.py --question_labels "what,when,why,how" --tokenizer_args "tc_ta,true,false"
rem CALL python gen_questions.py --question_labels "what,when,why,how" --tokenizer_args "tc_tqa,true,false"
rem CALL python gen_questions.py --question_labels "what,when,why,how" --tokenizer_args "tca_tq,true,false"

CALL python fine_tune_t5qg.py --question_labels "why" --tokenizer_args "tc_tqa,true,false" --num_train_epochs "5"
CALL python gen_questions.py --question_labels "why" --tokenizer_args "tc_tqa,true,false"


