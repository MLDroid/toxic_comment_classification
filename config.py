import os
import psutil
import torch

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
VALIDATION_SET_RATIO = 0.2

MULTIGPU = True if torch.cuda.device_count() > 1 else False

TRAINED_MODEL_FNAME_PREFIX = 'distilbert_toxic_comment'
TRAINED_MODEL_FNAME = None



