import torch

BERT_MODEL = 'bert-base-uncased'
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DATASET_ROOT= 'exp/data/'
RESULT_ROOT = "exp/result_bert_base_uncased"
METHOD = 'pref_ordered'

EPOCHS = 1
BATCH_SIZE = 12
MAX_ITERATION = 10000
LR = 1e-5