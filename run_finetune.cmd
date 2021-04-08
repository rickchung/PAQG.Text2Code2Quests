CALL conda activate depy3
CALL python fine_tune_t5qg.py --question_labels "what,when,why,how" --tokenizer_args "tc_ta,true,false"
CALL python fine_tune_t5qg.py --question_labels "what,when,why,how" --tokenizer_args "tc_tqa,true,false"
CALL python fine_tune_t5qg.py --question_labels "what,when,why,how" --tokenizer_args "tc_tq,true,false"

