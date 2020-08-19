import torch.nn as nn
from transformers import AutoModel

import config


class bert_classifier(nn.Module):
    def __init__(self, freeze_bert=True):
        super(bert_classifier, self).__init__()
        # Instantiating BERT model object
        self.model_name = config.MODEL_NAME
        self.bert_layer = AutoModel.from_pretrained(self.model_name)
        print(f'Loaded model from AutoMdodel for model name: {self.model_name}')

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.fc_layers = nn.ModuleList([nn.Linear(768, 2) for _ in range(6)])

        #Dropout
        self.dp20 = nn.Dropout(0.2)

    def forward(self, seq, attn_masks):
        if self.model_name.startswith('distil') or self.model_name.startswith('roberta')\
                or self.model_name.startswith('albert'):
            # print('calling: forward_distilbert')
            return self.forward_distilbert(seq,attn_masks)
        else:
            # print('calling forward_bert')
            return self.forward_bert(seq, attn_masks)


    def forward_bert(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        # Feeding the input to BERT model to obtain contextualized representations
        last_hidden_states, pooled_op = self.bert_layer(seq, attention_mask=attn_masks)

        # Obtaining the representation of [CLS] head
        # op = last_hidden_states.mean(1) #mean of all last hidden states
        op = pooled_op
        op = self.dp20(op)

        # Feeding op to the classifier layer
        logits_per_layer = []
        for fc in self.fc_layers:
            logits = fc(op)
            logits_per_layer.append(logits)

        return logits_per_layer


    def forward_distilbert(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        # Feeding the input to DistilBERT model to obtain contextualized representations
        cont_reps = self.bert_layer(seq, attention_mask=attn_masks)

        # Obtaining the representation of [CLS] head
        # cls_rep = cont_reps[0][:,0,:]
        op = cont_reps[0].mean(1)
        op = self.dp20(op)

        # Feeding op to the classifier layer
        logits_per_layer = []
        for fc in self.fc_layers:
            logits = fc(op)
            logits_per_layer.append(logits)


        return logits_per_layer