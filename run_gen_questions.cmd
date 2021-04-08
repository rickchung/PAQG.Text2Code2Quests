CALL conda activate depy3
CALL python gen_questions.py --model "tuned_what-how-which-where-who-why-other_cfTrue_qfFalse_t5-small" --output "cft_qff"
CALL python gen_questions.py --model "tuned_what-how-which-where-who-why-other_cfTrue_qfTrue_t5-small" --output "cft_qft"