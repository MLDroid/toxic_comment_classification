import os
import psutil
import torch
import sys

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

MODEL_NAME =  sys.argv[1]
# MODEL_NAME =  'bert-base-uncased'
# MODEL_NAME = 'roberta-base'

data_folder = './data'
train_fname = data_folder + '/train.csv'
test_fname = data_folder + '/test.csv'

MAX_SEQ_LEN = 512
NUM_EPOCHS = 10
BATCH_SIZE = 200
LR = 3e-4
NUM_CPU_WORKERS = psutil.cpu_count()
PRINT_EVERY = 100
BERT_LAYER_FREEZE = True

SAMPLE_RATIO = None
VALIDATION_SET_RATIO = 0.3

MULTIGPU = True if torch.cuda.device_count() > 1 else False

TRAINED_MODEL_FNAME_PREFIX = MODEL_NAME.upper()+'_toxic_comment_model'
TRAINED_MODEL_FNAME = None



