CALL conda activate depy3
CALL python fine_tune_t5qg.py --context_qtype_prefix true --question_qtype_prefix true
CALL python fine_tune_t5qg.py --context_qtype_prefix true --question_qtype_prefix false
rem CALL python fine_tune_t5qg.py --context_qtype_prefix false --question_qtype_prefix true
rem CALL python fine_tune_t5qg.py --context_qtype_prefix false --question_qtype_prefix false
