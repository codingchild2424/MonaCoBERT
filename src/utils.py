import pandas as pd
import numpy as np
import csv

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from torch.optim import SGD, Adam

from torch.nn.functional import binary_cross_entropy

import matplotlib.pyplot as plt

# collate_fn
def collate_fn(batch, pad_val=-1):

    q_seqs = []
    r_seqs = []

    for q_seq, r_seq in batch:

        q_seqs.append(torch.Tensor(q_seq)) 
        r_seqs.append(torch.Tensor(r_seq))

    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )

    mask_seqs = (q_seqs != pad_val)

    q_seqs, r_seqs = q_seqs * mask_seqs, r_seqs * mask_seqs

    return q_seqs, r_seqs, mask_seqs
    #|q_seqs| = (batch_size, maximum_sequence_length_in_the_batch)
    #|r_seqs| = (batch_size, maximum_sequence_length_in_the_batch)
    #|mask_seqs| = (batch_size, maximum_sequence_length_in_the_batch)

# for pid
def pid_collate_fn(batch, pad_val=-1):

    q_seqs = []
    r_seqs = []
    pid_seqs = []

    for q_seq, r_seq, pid_seq in batch:

        q_seqs.append(torch.Tensor(q_seq)) 
        r_seqs.append(torch.Tensor(r_seq)) 
        pid_seqs.append(torch.Tensor(pid_seq)) 

    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    pid_seqs = pad_sequence(
        pid_seqs, batch_first=True, padding_value=pad_val
    )

    mask_seqs = (q_seqs != pad_val)

    q_seqs, r_seqs, pid_seqs = q_seqs * mask_seqs, r_seqs * mask_seqs, pid_seqs * mask_seqs

    return q_seqs, r_seqs, pid_seqs, mask_seqs
    #|q_seqs| = (batch_size, max_seq_len)
    #|r_seqs| = (batch_size, max_seq_len)
    #|pid_seqs| = (batch_size, max_seq_len)
    #|mask_seqs| = (batch_size, max_seq_len)

# for pid_diff
def pid_diff_collate_fn(batch, pad_val=-1):

    q_seqs = []
    r_seqs = []
    pid_seqs = []
    diff_seqs = []

    for q_seq, r_seq, pid_seq, diff_seq in batch:

        q_seqs.append(torch.Tensor(q_seq))
        r_seqs.append(torch.Tensor(r_seq))
        pid_seqs.append(torch.Tensor(pid_seq))
        diff_seqs.append(torch.Tensor(diff_seq))

    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    pid_seqs = pad_sequence(
        pid_seqs, batch_first=True, padding_value=pad_val
    )
    diff_seqs = pad_sequence(
        diff_seqs, batch_first=True, padding_value=pad_val
    )

    mask_seqs = (q_seqs != pad_val)

    q_seqs, r_seqs, pid_seqs, diff_seqs = q_seqs * mask_seqs, r_seqs * mask_seqs, pid_seqs * mask_seqs, diff_seqs * mask_seqs

    return q_seqs, r_seqs, pid_seqs, diff_seqs, mask_seqs
    #|q_seqs| = (batch_size, max_seq_len)
    #|r_seqs| = (batch_size, max_seq_len)
    #|pid_seqs| = (batch_size, max_seq_len)
    #|diff_seqs| = (batch_size, max_seq_len)
    #|mask_seqs| = (batch_size, max_seq_len)

# get_optimizer
def get_optimizers(model, config):
    if config.optimizer == "adam":
        optimizer = Adam(model.parameters(), config.learning_rate)
    elif config.optimizer == "SGD":
        optimizer = SGD(model.parameters(), config.learning_rate)
    else:
        print("Wrong optimizer was used...")

    return optimizer

# get_crit
def get_crits(config):
    if config.crit == "binary_cross_entropy":
        crit = binary_cross_entropy
    elif config.crit == "rmse":
        class RMSELoss(nn.Module):
            def __init__(self, eps=1e-8):
                super().__init__()
                self.mse = nn.MSELoss()
                self.eps = eps
            def forward(self, y_hat, y):
                loss =  torch.sqrt(self.mse(y_hat, y) + self.eps)
                return loss
        crit = RMSELoss()
    else:
        print("Wrong criterion was used...")

    return crit

# early stop
class EarlyStopping:
    def __init__(self, metric_name, best_score=0, patience=10, verbose=True, delta=0, path='../checkpoints/checkpoint.pt'):
        self.metric_name = metric_name
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = best_score
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = val_loss
        
        # AUC has to be increased
        if self.metric_name == "AUC":
            # if there are no best_score
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
        # RMSE has to be decreased
        elif self.metric_name == "RMSE":
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score > self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss was updated ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def grp_range(a):
    count = np.unique(a,return_counts=1)[1]

    idx = count.cumsum()
    id_arr = np.ones(idx[-1],dtype=int)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -count[:-1]+1
    out = id_arr.cumsum()[np.argsort(a).argsort()]
    return out

#recoder
def recorder(test_auc_score, record_time, config):

    dir_path = "../score_records/"
    record_path = dir_path + "auc_record.csv"

    append_list = []

    append_list.append(record_time)
    append_list.extend([
        config.model_fn, config.batch_size, config.n_epochs,
        config.learning_rate, config.model_name, config.optimizer,
        config.dataset_name, config.max_seq_len, config.num_encoder,
        config.hidden_size, config.num_head, config.dropout_p,
        config.grad_acc, config.grad_acc_iter, config.fivefold, config.use_leakyrelu
    ])
    append_list.append("test_auc_score")
    append_list.append(test_auc_score)

    with open(record_path, 'a', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(append_list)


# visualizer
def visualizer(train_auc_scores, valid_auc_scores, record_time):
    plt.plot(train_auc_scores)
    plt.plot(valid_auc_scores)
    plt.legend(['train_auc_scores', 'valid_auc_scores'])
    path = "../graphs/"
    plt.savefig(path + record_time + ".png")
