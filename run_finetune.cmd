CALL conda activate depy3
rem CALL python fine_tune_t5qg.py --question_labels "what,when,why,how" --tokenizer_args "tc_ta,true,false"
rem CALL python fine_tune_t5qg.py --question_labels "what,when,why,how" --tokenizer_args "tc_tqa,true,false"
rem CALL python fine_tune_t5qg.py --question_labels "what,when,why,how" --tokenizer_args "tc_tq,true,false"

rem CALL python fine_tune_t5qg.py --question_labels "what,why" --tokenizer_args "tc_ta,true,true"
CALL python gen_questions.py --question_labels "what,why" --tokenizer_args "tc_ta,true,true" --output "tc_ta"

