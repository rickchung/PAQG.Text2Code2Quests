import datasets
import transformers

from utils import *


class QGDataProcessor:
    """
    A data process is responsible for loading a dataset and prepare the input for a downstream task.
    """
    # These tokens are for T5-based models only
    SPECIAL_TOKENS = {
        't5': ['<hl>']
    }

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, base_model_type='t5', max_source_len=512,
                 max_target_len=32):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

        # Get the special tokens defined for `base_model_type`
        self.hl_token = self.SPECIAL_TOKENS[base_model_type][0]

        # Add custom tokens to the tokenizer
        self.tokenizer.add_tokens([self.hl_token])

        # Get a logger for this class
        self.logger = get_my_logger(self.__class__.__name__)

    def load_dataset(self, dataset_name='squad'):
        """
        Load the dataset called `dataset_name` via HF Datasets. In this study, we use SQuAD.
        See https://huggingface.co/datasets/squad for more detail about this dataset.
        """
        # Load the dataset via HF Datasets
        dataset = datasets.load_dataset(dataset_name)

        # Show the basic information about the dataset
        self.logger.info(f"Dataset Description: {dataset['train'].description}")
        # self.logger.info(f"Dataset Citation: {dataset['train'].citation}")
        # self.logger.info(f"Dataset Homepage: {dataset['train'].homepage}")
        # self.logger.info(f"Dataset License: {dataset['train'].license}")
        # self.logger.info(f"Dataset Summary: {dataset}")

        # Reserve the dataset for later processing
        self.dataset = dataset

        # Process the dataset
        self._preprocess_question_types()

    def get_tokenized_data(self, context_prefix=False, quest_prefix=False, highlight_answer=False):
        """
        Prefix the "context" and "question", and tokenize the dataset `self.dataset` by the tokenizer `self.tokenizer`.
        """

        def _tokenize(item):
            # Tokenize an `item` by the tokenizer
            """
            A helper function that tokenizes an input item by a T5-based tokenizer.
            """
            common_args = {'padding': 'max_length', 'pad_to_max_length': True, 'truncation': True}

            context = item['context']
            question = item['question']
            question_type = item['quest_type']

            if context_prefix:
                # Add the question type to the context as prefix to inform the model
                context = f'{question_type}: {context}'
            if quest_prefix:
                # TODO: Does the question need the prefix?
                question = f'{question_type}: {question}'

            source_tokens = self.tokenizer(context, max_length=self.max_source_len, **common_args)
            target_tokens = self.tokenizer(question, max_length=self.max_target_len, **common_args)

            # Store the tokens in the pre-defined column names required by the base model (T5).
            # Note: If you'd like to use a different model, you may have to replace these
            # names by those required by the model.
            encodings = {
                'input_ids': source_tokens['input_ids'],
                'labels': target_tokens['input_ids'],
                'attention_mask': source_tokens['attention_mask'],
                'source_text': context,
                'target_text': question,
            }

            return encodings

        return self.dataset.map(_tokenize)

    def _preprocess_question_types(self):
        """
        Process the loaded dataset in `load_dataset()`. This method does the following tasks:
        - Add one extra column `question_type` to denote the type of questions
        """

        def _get_question_type(item):
            """
            A helper function that decides the question type according to the question text in `item`.
            """
            # Filter the question text and assign a new question type when anything matches
            question_words = ["what", "who", "where", "when", "which", "why", "how"]
            quest_type = 'other'
            lowered_question = item['question'].lower()
            for k in question_words:
                if lowered_question.startswith(k):
                    quest_type = k
                    break;
                # if k in lowered_question:
                #     quest_type = k
                #     break

            return {'quest_type': quest_type}

        # Assign the question type to each item
        self.dataset = self.dataset.map(_get_question_type)
