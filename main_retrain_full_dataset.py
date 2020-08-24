import torch
import numpy as np
import time
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score
import pandas as pd

import config
import dataset
import model

def get_auc_from_logits(logits, labels):
    all_preds = np.array([logits[i].argmax(1).cpu().numpy() for i in range(6)]).T
    all_trues = labels
    aucs_for_samples = np.array([safe_calc_auc(all_trues[:, col], all_preds[:, col]) for col in range(6)])
    return aucs_for_samples


def safe_calc_auc(trues, preds):
    try:
        auc = roc_auc_score(trues, preds)
    except:
        auc = 0.5
    return auc


def train_model(net, criterion, optimizer, scheduler, train_loader,
                print_every=100, n_epochs=10, device='cpu', save_model=True,
                start_training_epoch_at=1):

    for e in range(start_training_epoch_at, start_training_epoch_at+n_epochs+1):
        t0 = time.perf_counter()
        e_loss = []
        for batch_num, (seq_attnmask_labels) in enumerate(tqdm(train_loader), start=1):
            # Clear gradients
            optimizer.zero_grad()

            #get the 3 input args for this batch
            seq, attn_mask, labels = seq_attnmask_labels

            # Converting these to cuda tensors
            seq, attn_mask, labels = seq.cuda(device), attn_mask.cuda(device), labels.cuda(device)

            # Obtaining the logits from the model
            logits = net(seq, attn_mask)

            # Computing loss
            loss_all_classes = 0.
            for class_id in range(6):
                pred = logits[class_id]
                true = labels[:,class_id]
                loss_for_this_class = criterion[class_id](pred, true)
                loss_all_classes += loss_for_this_class
            e_loss.append(loss_all_classes.item())

            # Backpropagating the gradients for losses on all classes
            loss_all_classes.backward()

            # Optimization step
            optimizer.step()
            scheduler.step()

            if (batch_num + 1) % print_every == 0:
                auc = get_auc_from_logits(logits, labels.cpu().numpy())
                print(f"batch {batch_num+1} of epoch {e} complete. Loss : {loss_all_classes.item()} "
                      f"AUC : {auc.round(2)}")

        t = time.perf_counter() - t0
        e_loss = np.array(e_loss).mean()
        print(f'Done epoch: {e} in {round(t,2)} sec, epoch loss: {e_loss}')


        if save_model:
            save_fname = f'full_training_set_{config.TRAINED_MODEL_FNAME_PREFIX}_e_{e}.pt'
            torch.save(net, save_fname)
            print(f'Saved model at: {save_fname} after epoch: {e}')




def compute_class_weight_balanced(y):
    n_samples = len(y)
    n_classes = 2
    w = n_samples / (n_classes * np.bincount(y))
    return w


def get_class_weigts(train_df, max_class_weight=50.):
    class_weights_dict = {}
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_df = train_df[label_cols]
    labels = np.array(train_df, dtype=int)
    for col_index in range(labels.shape[1]):
        w = compute_class_weight_balanced(labels[:,col_index])
        w = w.clip(max = max_class_weight)
        class_weights_dict[col_index] = w
    return class_weights_dict


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Creating instances of training and validation set
    full_train_df = pd.read_csv(config.train_fname)
    # full_train_df = full_train_df.sample(frac=.01)

    #for dataset loader
    train_set = dataset.dataset(full_train_df, max_len=config.MAX_SEQ_LEN)

    #creating dataloader
    train_loader = DataLoader(train_set, shuffle = True,
                              batch_size=config.BATCH_SIZE, num_workers=config.NUM_CPU_WORKERS)

    #creating BERT model
    if config.TRAINED_MODEL_FNAME:
        bert_model = torch.load(config.TRAINED_MODEL_FNAME)
        print(f'Loaded trained model: {bert_model} from file: {config.TRAINED_MODEL_FNAME}')
    else:
        bert_model = model.bert_classifier(freeze_bert=config.BERT_LAYER_FREEZE)
        print(f"created NEW BERT model for finetuning: {bert_model}")
        bert_model.cuda()

        # Multi GPU setting
        if config.MULTIGPU:
            bert_model = nn.DataParallel(bert_model, device_ids=[0, 1, 2, 3])


    #loss function
    class_weights_dict = get_class_weigts(full_train_df)
    class_weights_dict = {c: torch.tensor(w).to(device).float() for c, w in class_weights_dict.items()}
    criterion = [nn.CrossEntropyLoss(weight=class_weights_dict[c]) for c in sorted(class_weights_dict.keys())]

    #optimizer and scheduler
    param_optimizer = list(bert_model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_set) / config.BATCH_SIZE * config.NUM_EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=config.LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    train_model(bert_model, criterion, optimizer, scheduler, train_loader,
                print_every=config.PRINT_EVERY, n_epochs=config.NUM_EPOCHS, device=device,
                start_training_epoch_at=1)


if __name__ == '__main__':
    main()