import torch

BERT_MODEL = 'bert-base-uncased'
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DATASET_ROOT= 'exp/data/'
RESULT_ROOT = "exp/result_bert_base_uncased"
METHOD = 'pref_ordered'

EPOCHS = 2
BATCH_SIZE = 8
MAX_ITERATION = 15000
LR = 1e-5