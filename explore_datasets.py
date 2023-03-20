import datasets
import numpy as np

squad = datasets.load_dataset('squad').flatten()
squad_train = squad['train']
squad_valid = squad['validation']

print('Train\t\t\t'
      'Validation')

# Size
print(f'Num of QA pairs\n'
      f'{len(squad_train)}\t\t\t'
      f'{len(squad_valid)}')

# How many contexts are there?

train_context = squad_train['context']
valid_context = squad_valid['context']

tc_unique, tc_count = np.unique(train_context, return_counts=True)
vc_unique, vc_count = np.unique(valid_context, return_counts=True)

print(f'Num of unique context passages\n'
      f'{len(tc_unique)}\t\t\t'
      f'{len(vc_unique)}')
print(f'Number of QA pairs per context\n'
      f'M={tc_count.mean():.2f}; SD={tc_count.std():.2f}\t\t\t'
      f'M={vc_count.mean():.2f}; SD={vc_count.std():.2f}')

# How many questions are there? What is the distribution of questions for each context?

# What are the characteristics of questions?

