import numpy as np
import pandas as pd

from torch.utils.data import Dataset

DATASET_DIR = "../datasets/algebra05/preprocessed_df.csv"

class ALGEBRA2005(Dataset):
    def __init__(self, max_seq_len, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        
        # using self.preprocess for making sequences
        self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.r_list, self.q2idx, \
            self.u2idx = self.preprocess() 

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        self.num_r = self.r_list.shape[0]

        # using self.match_seq_len for making same length of sequences; the sequence length will be max_seq_len
        self.q_seqs, self.r_seqs = \
            self.match_seq_len(self.q_seqs, self.r_seqs, max_seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_dir, sep='\t').sort_values(by=["timestamp"])

        u_list = np.unique(df["user_id"].values) # unique user_id
        q_list = np.unique(df["skill_id"].values) # unique concept(skill_id)
        r_list = np.unique(df["correct"].values) # unique correct

        u2idx = {u: idx for idx, u in enumerate(u_list)} # dict for user
        q2idx = {q: idx for idx, q in enumerate(q_list)} # dict for concept

        q_seqs = [] # list of concepts
        r_seqs = [] # list of responses

        for u in u_list:
            df_u = df[df["user_id"] == u]

            q_seq = np.array([q2idx[q] for q in df_u["skill_id"].values])
            r_seq = df_u["correct"].values

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)

        return q_seqs, r_seqs, q_list, u_list, r_list, q2idx, u2idx

    # the sequence length will be the same as max_seq_len
    def match_seq_len(self, q_seqs, r_seqs, max_seq_len, pad_val=-1):

        proc_q_seqs = []
        proc_r_seqs = []

        for q_seq, r_seq in zip(q_seqs, r_seqs):

            i = 0
            while i + max_seq_len < len(q_seq):
                proc_q_seqs.append(q_seq[i:i + max_seq_len])
                proc_r_seqs.append(r_seq[i:i + max_seq_len])

                i += max_seq_len

            proc_q_seqs.append(
                np.concatenate(
                    [
                        q_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
            proc_r_seqs.append(
                np.concatenate(
                    [
                        r_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )

        return proc_q_seqs, proc_r_seqs