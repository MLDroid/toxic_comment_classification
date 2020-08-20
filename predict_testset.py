import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from copy import deepcopy

import config
import dataset


def test(net, test_loader, device='cpu'):
    sm = torch.nn.Softmax(dim=1)
    net.eval()
    with torch.no_grad():
        for batch_num, (seq, attn_masks) in enumerate(tqdm(test_loader), start=1):
            seq, attn_masks = seq.to(device), attn_masks.to(device)
            logits = net(seq, attn_masks)
            logits = [sm(i) for i in logits]
            preds = np.array([logits[i][:,1].cpu().numpy() for i in range(6)]).T
            if batch_num == 1:
                all_preds = preds
            else:
                all_preds = np.vstack([all_preds, preds])
    return all_preds


def get_sub_df(full_test_df, all_preds):
    sub_df = deepcopy(full_test_df)
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for i,c in zip(range(6),categories):
        preds = all_preds[:,i]
        sub_df[c] = preds
    return sub_df


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #for now!!!
    device = 'cpu'

    # Creating instances of training and validation set
    full_test_df = pd.read_csv(config.test_fname)
    # full_test_df = full_test_df.sample(frac=.0001)

    #for dataset loader
    full_test_set = dataset.test_dataset(full_test_df, max_len=config.MAX_SEQ_LEN)


    #creating dataloader
    #!!! MAKE SURE THE DATASET IS NOT SHUFFLED !!!
    full_test_ds_loader = DataLoader(full_test_set, shuffle=False,
                                     batch_size=config.BATCH_SIZE, num_workers=config.NUM_CPU_WORKERS)

    #Loading a trained model
    bert_model = torch.load(config.TRAINED_MODEL_FNAME)
    torch.load(config.TRAINED_MODEL_FNAME, map_location='cpu')
    print(f'Loaded trained model: {bert_model} from file: {config.TRAINED_MODEL_FNAME}')
    # bert_model.to(device)

    # Multi GPU setting
    # if config.MULTIGPU and use_cuda:
    #     bert_model = nn.DataParallel(bert_model,  device_ids=[0, 1, 2, 3])

    all_preds = test(bert_model, test_loader = full_test_ds_loader, device=device)

    sub_df = get_sub_df(full_test_df, all_preds)

    save_fname = f'{config.TRAINED_MODEL_FNAME}_preds.csv'
    sub_df.to_csv(save_fname,index=False)

    print(f'model predictions saved to {save_fname}')


if __name__ == '__main__':
    main()