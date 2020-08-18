import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, DistilBertTokenizerFast
import pandas as pd
import numpy as np
from tqdm import tqdm

import config


def get_train_valid_df(dataset_fname, sample_ratio=None, valid_ratio=0.2):
    df = pd.read_csv(dataset_fname)
    if sample_ratio:
        df = df.sample(frac=sample_ratio)
    print(f'Loaded dataframe of shape: {df.shape} from {dataset_fname}')

    train_df = df.sample(frac=1-valid_ratio)
    valid_df = df[~df.index.isin(train_df.index)]

    return train_df, valid_df



class dataset(Dataset):
    def __init__(self, df, max_len):
        # Initialize the BERT tokenizer
        # self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.max_len = max_len

        self.sent_token_ids_attn_masks = [self._get_token_ids_attn_mask(s) for s in tqdm(df.comment_text)]
        self.labels = self._get_tc_dataset_labels(df)

        print(f'Loaded X_train and y_train, shapes: {len(self.sent_token_ids_attn_masks), self.labels.shape}')


    def _get_tc_dataset_labels(self, df):
        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        df = df[label_cols]
        labels = np.array(df,dtype=int)
        return labels


    def _get_token_ids_attn_mask(self, sentence):
        sentence = sentence.lower().strip()
        tokens = self.tokenizer.tokenize(sentence)  # Tokenize the sentence
        tokens = ['[CLS]'] + tokens + ['[SEP]']  # Insering the CLS and SEP token in the beginning and end of the sentence
        if len(tokens) < self.max_len:
            tokens = tokens + ['[PAD]' for _ in range(self.max_len - len(tokens))]  # Padding sentences
        else:
            tokens = tokens[:self.max_len - 1] + ['[SEP]']  # Pruning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(
            tokens)  # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids)  # Converting the list to a pytorch tensor

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        #Selecting the sentence and label at the specified index in the data frame
        token_ids,attn_mask = self.sent_token_ids_attn_masks[index] #list index
        label = self.labels[index] #array index

        return token_ids, attn_mask, label
