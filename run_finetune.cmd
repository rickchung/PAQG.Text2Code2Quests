CALL conda activate depy3

CALL python fine_tune_t5qg.py --question_labels "what,when,why,how" --tokenizer_args "tc_tq,true,true"
CALL python gen_questions.py --question_labels "what,when,why,how" --tokenizer_args "tc_tq,true,true"

CALL python fine_tune_t5qg.py --question_labels "what,when,why,how" --tokenizer_args "tc_ta,true,true"
CALL python gen_questions.py --question_labels "what,when,why,how" --tokenizer_args "tc_ta,true,true"

CALL python fine_tune_t5qg.py --question_labels "what,when,why,how" --tokenizer_args "tc_tqa,true,true"
CALL python gen_questions.py --question_labels "what,when,why,how" --tokenizer_args "tc_tqa,true,true"

CALL python fine_tune_t5qg.py --question_labels "what,when,why,how" --tokenizer_args "tca_tq,true,true"
CALL python gen_questions.py --question_labels "what,when,why,how" --tokenizer_args "tca_tq,true,true"


