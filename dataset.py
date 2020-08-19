import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, DistilBertTokenizerFast, RobertaTokenizerFast
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import config


def get_train_valid_df(dataset_fname, sample_ratio=None, valid_ratio=0.2, save_dfs=True):
    df = pd.read_csv(dataset_fname)
    if sample_ratio:
        df = df.sample(frac=sample_ratio)
    print(f'Loaded dataframe of shape: {df.shape} from {dataset_fname}')

    # train_df = df.sample(frac=1-valid_ratio)
    # valid_df = df[~df.index.isin(train_df.index)]

    train_df, valid_df = train_test_split(df, random_state=42, test_size=valid_ratio, shuffle=True)

    if save_dfs:
        model_name = config.MODEL_NAME.upper()
        train_df.to_csv(f'{model_name}_train_df_.csv',index=False)
        valid_df.to_csv(f'{model_name}_valid_df_.csv',index=False)

    return train_df, valid_df



class dataset(Dataset):
    def __init__(self, df, max_len):
        self.model_name = config.MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        print(f'Using tokenizer: {self.tokenizer} for model: {self.model_name}')

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
        sentence = ' '.join(sentence.split())

        # tokens = self.tokenizer.tokenize(sentence)  # Tokenize the sentence
        # tokens = ['[CLS]'] + tokens + ['[SEP]']  # Insering the CLS and SEP token in the beginning and end of the sentence
        # if len(tokens) < self.max_len:
        #     tokens = tokens + ['[PAD]' for _ in range(self.max_len - len(tokens))]  # Padding sentences
        # else:
        #     tokens = tokens[:self.max_len - 1] + ['[SEP]']  # Pruning the list to be of specified max length
        #
        # tokens_ids = self.tokenizer.convert_tokens_to_ids(
        #     tokens)  # Obtaining the indices of the tokens in the BERT Vocabulary
        # tokens_ids_tensor = torch.tensor(tokens_ids)  # Converting the list to a pytorch tensor
        # # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        # attn_mask = (tokens_ids_tensor != 0).long()
        # return tokens_ids_tensor, attn_mask

        inputs = self.tokenizer.encode_plus(sentence, None,
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            pad_to_max_length=True,
                                            truncation=True
                                            )
        tokens_ids_tensor = torch.tensor(inputs["input_ids"], dtype=torch.long)
        attn_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        return tokens_ids_tensor, attn_mask


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        #Selecting the sentence and label at the specified index in the data frame
        token_ids,attn_mask = self.sent_token_ids_attn_masks[index] #list index
        label = self.labels[index] #array index
        return token_ids, attn_mask, label



# class test_dataset(Dataset):
#     def __init__(self, df, max_len):
#         self.model_name = config.MODEL_NAME
#         # Initialize the BERT tokenizer
#         if self.model_name.startswith('distilbert'):
#             self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
#         elif self.model_name.startswith('roberta'):
#             self.tokenizer = RobertaTokenizerFast.from_pretrained(self.model_name)
#         else:
#             self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
#         print(f'Using tokenizer: {self.tokenizer} for model: {self.model_name}')
#
#         self.max_len = max_len
#
#         self.sent_token_ids_attn_masks = [self._get_token_ids_attn_mask(s) for s in tqdm(df.comment_text)]
#
#         print(f'Loaded X_test shape: {len(self.sent_token_ids_attn_masks)}')
#
#
#     def _get_token_ids_attn_mask(self, sentence):
#         sentence = sentence.lower().strip()
#         tokens = self.tokenizer.tokenize(sentence)  # Tokenize the sentence
#         tokens = ['[CLS]'] + tokens + ['[SEP]']  # Insering the CLS and SEP token in the beginning and end of the sentence
#         if len(tokens) < self.max_len:
#             tokens = tokens + ['[PAD]' for _ in range(self.max_len - len(tokens))]  # Padding sentences
#         else:
#             tokens = tokens[:self.max_len - 1] + ['[SEP]']  # Pruning the list to be of specified max length
#
#         tokens_ids = self.tokenizer.convert_tokens_to_ids(
#             tokens)  # Obtaining the indices of the tokens in the BERT Vocabulary
#         tokens_ids_tensor = torch.tensor(tokens_ids)  # Converting the list to a pytorch tensor
#
#         # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
#         attn_mask = (tokens_ids_tensor != 0).long()
#         return tokens_ids_tensor, attn_mask
#
#
#     def __len__(self):
#         return len(self.sent_token_ids_attn_masks)
#
#
#     def __getitem__(self, index):
#         #Selecting the sentence at the specified index in the data frame
#         token_ids,attn_mask = self.sent_token_ids_attn_masks[index] #list index
#         return token_ids, attn_mask
