import os
import psutil
import torch

data_folder = '/home/anna/toxic_comment_classification/data'
train_fname = data_folder + '/train.csv'
test_fname = data_folder + '/test.csv'

MAX_SEQ_LEN = 512
NUM_EPOCHS = 20
BATCH_SIZE = 100
LR = 3e-4
NUM_CPU_WORKERS = psutil.cpu_count()
PRINT_EVERY = 100
BERT_LAYER_FREEZE = True

SAMPLE_RATIO = 0.2
VALIDATION_SET_RATIO = 0.2

MULTIGPU = True if torch.cuda.device_count() > 1 else False

