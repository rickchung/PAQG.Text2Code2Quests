CALL conda activate depy3
CALL python gen_questions.py --question_labels "what,why" --tokenizer_args "tc_ta,true,true" --output "tc_ta"
rem CALL python gen_questions.py --question_labels "what,why,when,how" --tokenizer_args "tc_tq,true,false" --output "tc_tq"
rem CALL python gen_questions.py --question_labels "what,why,when,how" --tokenizer_args "tc_tqa,true,false" --output "tc_tqa"
