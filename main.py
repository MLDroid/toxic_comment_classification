import torch
import numpy as np
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score


import config
import dataset
import model

def get_auc_from_logits(logits, labels):
    auc_for_all_classes = np.zeros(6)
    preds = np.zeros(labels.shape)
    for class_id in range(6):
        pred_class = logits[class_id]
        pred_class = pred_class.argmax(1)
        preds[:, class_id] = pred_class.cpu().numpy()
        try:
            auc = roc_auc_score(labels[:, class_id], preds[:, class_id])
        except ValueError:
            auc = 0.5

        auc_for_all_classes[class_id] = auc
    return auc_for_all_classes


def test(net, test_loader, device='cpu'):
    aucs_for_samples = []
    net.eval()
    with torch.no_grad():
        for (seq, attn_masks, labels) in tqdm(test_loader):

            seq, attn_masks = seq.cuda(device), attn_masks.cuda(device)
            logits = net(seq, attn_masks)
            auc_for_all_classes = get_auc_from_logits(logits, labels)
            aucs_for_samples.append(auc_for_all_classes)

    aucs_for_samples = np.array(aucs_for_samples).mean(0)
    return aucs_for_samples


def train_model(net, criterion, optimizer, scheduler, train_loader, test_loader=None, print_every=100, n_epochs=10, device='cpu'):
    for e in range(1, n_epochs+1):
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
            loss = 0.
            for class_id in range(6):
                pred = logits[class_id]
                true = labels[:,class_id]
                loss += criterion(pred, true.long())
            e_loss.append(loss.item())

            # Backpropagating the gradients
            loss.backward()

            # Optimization step
            optimizer.step()
            scheduler.step()

            if (batch_num + 1) % print_every == 0:
                acc = get_auc_from_logits(logits, labels)
                print(f"batch {batch_num+1} of epoch {e} complete. Loss : {loss.item()} AUC : {acc}")

        t = time.perf_counter() - t0
        e_loss = np.array(e_loss).mean()
        print(f'Done epoch: {e} in {round(t,2)} sec, epoch loss: {e_loss}')
        if test_loader != None:
            test_acc = test(net, test_loader, device=device)
            print(f'After epoch: {e}, test AUC: {test_acc.round(2)}')



def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Creating instances of training and validation set
    train_df, valid_df = dataset.get_train_valid_df(dataset_fname = config.train_fname,
                                                    sample_ratio=config.SAMPLE_RATIO,
                                                    valid_ratio=config.VALIDATION_SET_RATIO)

    #for dataset loader
    train_set = dataset.dataset(train_df, max_len=config.MAX_SEQ_LEN)
    valid_set = dataset.dataset(valid_df, max_len=config.MAX_SEQ_LEN)

    #creating dataloader
    train_loader = DataLoader(train_set, shuffle = True,
                              batch_size=config.BATCH_SIZE,
                              num_workers=config.NUM_CPU_WORKERS)
    test_loader = DataLoader(valid_set, shuffle = True,
                             batch_size=config.BATCH_SIZE,
                             num_workers=config.NUM_CPU_WORKERS)

    #creating BERT model
    bert_model = model.bert_classifier(freeze_bert=config.BERT_LAYER_FREEZE)
    bert_model.cuda()

    #loss function
    criterion = nn.CrossEntropyLoss()

    #optimizer and scheduler
    # optimizer = optim.Adam(bert_model.parameters(), lr=config.LR)
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
    # print(optimizer_parameters)

    num_train_steps = int(len(train_set) / config.BATCH_SIZE * config.NUM_EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=config.LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    # Multi GPU setting
    if config.MULTIGPU:
        bert_model = nn.DataParallel(bert_model,  device_ids=[0, 1, 2, 3])


    print(f"created BERT model for finetuning: {bert_model}")

    aucs_for_samples = test(bert_model, test_loader, device=device)
    print(f'AUC before training: {aucs_for_samples.round(2)}')
    train_model(bert_model, criterion, optimizer, scheduler, train_loader, test_loader,
                print_every=config.PRINT_EVERY, n_epochs=config.NUM_EPOCHS, device=device)


if __name__ == '__main__':
    main()