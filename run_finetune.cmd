CALL conda activate depy3

CALL python fine_tune_t5qg.py --dry --question_labels "what,when,why,how" --tokenizer_args "tc_tq,true,false"
CALL python gen_questions.py --dry --question_labels "what,when,why,how" --tokenizer_args "tc_tq,true,false" --output "tc_tq_what_when_why_how"

CALL python fine_tune_t5qg.py --dry --question_labels "what,when,why,how" --tokenizer_args "tc_ta,true,false"
CALL python gen_questions.py --dry --question_labels "what,when,why,how" --tokenizer_args "tc_ta,true,false" --output "tc_ta_what_when_why_how"

CALL python fine_tune_t5qg.py --dry --question_labels "what,when,why,how" --tokenizer_args "tc_tqa,true,false"
CALL python gen_questions.py --dry --question_labels "what,when,why,how" --tokenizer_args "tc_tqa,true,false" --output "tc_tqa_what_when_why_how"

CALL python fine_tune_t5qg.py --dry --question_labels "what,when,why,how" --tokenizer_args "tca_tq,true,false"
CALL python gen_questions.py --dry --question_labels "what,when,why,how" --tokenizer_args "tca_tq,true,false" --output "tca_tq_what_when_why_how"


