import datasets
import transformers

from utils import *


class QgDataProcessor:
    """
    A data process is responsible for loading a dataset and prepare the input for a downstream task.
    """
    # These tokens are for T5-based models only
    SPECIAL_TOKENS = {
        't5': {'HLS': 'highlight:'}
    }
    path_processed_data = Path("models", "processed_qgdata")

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, base_model_type='t5', max_source_len=512,
                 max_target_len=32):
        self.dataset = None
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

        # Get the special tokens defined for `base_model_type`
        self.special_tokens = self.SPECIAL_TOKENS[base_model_type]
        # Add custom tokens to the tokenizer
        self.tokenizer.add_tokens(self.special_tokens.values())
        # Get a logger for this class
        self.logger = get_my_logger(self.__class__.__name__)

    def load_dataset(self, dataset_name='squad'):
        """
        Load the dataset called `dataset_name` via HF Datasets. In this study, we use SQuAD.
        See https://huggingface.co/datasets/squad for more detail about this dataset.
        """

        if self.path_processed_data.exists():
            self.logger.warn("Reuse an existing processed QA dataset")
            self.dataset = datasets.load_from_disk(str(self.path_processed_data))
        else:
            self.logger.info("Build and process a new QA dataset")
            # Load the dataset via HF Datasets
            dataset = datasets.load_dataset(dataset_name)

            # Show the basic information about the dataset
            self.logger.info(f'Base dataset: {dataset_name}')
            # self.logger.info(f"Dataset Description: {dataset['train'].description}")
            # self.logger.info(f"Dataset Citation: {dataset['train'].citation}")
            # self.logger.info(f"Dataset Homepage: {dataset['train'].homepage}")
            # self.logger.info(f"Dataset License: {dataset['train'].license}")
            # self.logger.info(f"Dataset Summary: {dataset}")

            # Reserve the dataset for later processing
            self.dataset = dataset
            # Process the dataset
            self._preprocess_question_types()
            self.dataset.save_to_disk(str(self.path_processed_data))

    def get_tokenized_tc_tqa(self, source_prefix, target_prefix):
        """
        Answer-aware QG: Typed Context -> Typed Question | Answer

        """
        source_prefix = (source_prefix.lower() == 'true')
        target_prefix = (target_prefix.lower() == 'true')

        def _tokenize(item):
            source = item['context']
            target = f'{item["question"]} {self.special_tokens["HLS"]} {item["answers"]["text"][0]}'
            question_type = item['quest_type']

            if source_prefix:
                source = f'{question_type}: {source}'
            if target_prefix:
                target = f'{question_type}: {target}'

            common_args = {'padding': 'max_length', 'pad_to_max_length': True, 'truncation': True}
            source_tokens = self.tokenizer(source, max_length=self.max_source_len, **common_args)
            target_tokens = self.tokenizer(target, max_length=self.max_target_len, **common_args)

            return {
                'input_ids': source_tokens['input_ids'],
                'labels': target_tokens['input_ids'],
                'attention_mask': source_tokens['attention_mask'],
                'source_text': source,
                'target_text': target,
            }

        return self.dataset.map(_tokenize)

    def get_tokenized_tca_tq(self, source_prefix, target_prefix):
        """
        Answer-aware QG: Typed Context | Answer -> Typed Question

        """
        source_prefix = (source_prefix.lower() == 'true')
        target_prefix = (target_prefix.lower() == 'true')

        def _tokenize(item):
            source = f'{item["context"]} {self.special_tokens["HLS"]} {item["answers"]["text"][0]}'
            target = item['question']
            question_type = item['quest_type']

            if source_prefix:
                source = f'{question_type}: {source}'
            if target_prefix:
                target = f'{question_type}: {target}'

            common_args = {'padding': 'max_length', 'pad_to_max_length': True, 'truncation': True}
            source_tokens = self.tokenizer(source, max_length=self.max_source_len, **common_args)
            target_tokens = self.tokenizer(target, max_length=self.max_target_len, **common_args)

            return {
                'input_ids': source_tokens['input_ids'],
                'labels': target_tokens['input_ids'],
                'attention_mask': source_tokens['attention_mask'],
                'source_text': source,
                'target_text': target,
            }

        return self.dataset.map(_tokenize)

    def get_tokenized_tc_tq(self, source_prefix, target_prefix):
        """
        QG: Typed Context -> Typed Question
        """
        source_prefix = (source_prefix.lower() == 'true')
        target_prefix = (target_prefix.lower() == 'true')

        def _tokenize(item):
            context = item['context']
            question = item['question']
            question_type = item['quest_type']

            if source_prefix:
                # Add the question type to the context as prefix to inform the model
                context = f'{question_type}: {context}'
            if target_prefix:
                # TODO: Does the question need the prefix?
                question = f'{question_type}: {question}'

            common_args = {'padding': 'max_length', 'pad_to_max_length': True, 'truncation': True}
            source_tokens = self.tokenizer(context, max_length=self.max_source_len, **common_args)
            target_tokens = self.tokenizer(question, max_length=self.max_target_len, **common_args)

            return {
                'input_ids': source_tokens['input_ids'],
                'labels': target_tokens['input_ids'],
                'attention_mask': source_tokens['attention_mask'],
                'source_text': context,
                'target_text': question,
            }

        return self.dataset.map(_tokenize)

    def get_tokenized_tc_ta(self, source_prefix, target_prefix):
        """
        KPE: Typed Context -> Typed Answer
        """
        source_prefix = (source_prefix.lower() == 'true')
        target_prefix = (target_prefix.lower() == 'true')

        def _tokenize(item):
            source, target = item['context'], item['answers']['text'][0]
            question_type = item['quest_type']

            if source_prefix:
                source = f'{question_type}: {source}'
            if target_prefix:
                target = f'{question_type}: {target}'

            common_args = {'padding': 'max_length', 'pad_to_max_length': True, 'truncation': True}
            source_tokens = self.tokenizer(source, max_length=self.max_source_len, **common_args)
            target_tokens = self.tokenizer(target, max_length=self.max_target_len, **common_args)

            return {
                'input_ids': source_tokens['input_ids'],
                'labels': target_tokens['input_ids'],
                'attention_mask': source_tokens['attention_mask'],
                'source_text': source,
                'target_text': target,
            }

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
                if k in lowered_question:
                    quest_type = k
                    break

            return {'quest_type': quest_type}

        # Assign the question type to each item
        self.dataset = self.dataset.map(_get_question_type)
