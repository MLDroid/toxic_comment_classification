import psutil
import torch
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MODEL_NAME =  sys.argv[1]
# MODEL_NAME = 'bert-base-uncased', 'roberta-base', 'distilbert-base-uncased'

data_folder = './data'
train_fname = data_folder + '/train.csv'
test_fname = data_folder + '/test.csv'

MAX_SEQ_LEN = 4096 if 'longformer' in MODEL_NAME else 512
NUM_EPOCHS = 10
BATCH_SIZE = 30
LR = 3e-4
NUM_CPU_WORKERS = psutil.cpu_count()
PRINT_EVERY = 100
BERT_LAYER_FREEZE = True

SAMPLE_RATIO = None #load only this % of the train.csv DF
VALIDATION_SET_RATIO = 0.3 #validation set extracted from training set

MULTIGPU = True if torch.cuda.device_count() > 1 else False #when using xlarge vs 16x large AWS m/c

#these 2 are not used yet
TRAINED_MODEL_FNAME_PREFIX = MODEL_NAME.upper()+'_toxic_comment_model'
TRAINED_MODEL_FNAME = MODEL_NAME.upper()+'_toxic_comment_model_e_10.pt'

START_TRAINING_EPOCH_AT = 11



