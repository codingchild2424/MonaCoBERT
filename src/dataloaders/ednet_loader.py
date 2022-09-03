import numpy as np
import pandas as pd
import os
import pickle

from torch.utils.data import Dataset

DATASET_DIR = "../datasets/EdNet/pre_train.csv"
PICKLE_DIR = "../datasets/EdNet/"

class EDNET(Dataset):
    def __init__(self, max_seq_len, dataset_dir=DATASET_DIR, pickle_dir=PICKLE_DIR) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.pickle_dir = pickle_dir
        
        if os.path.exists(os.path.join(self.pickle_dir, "q_seqs.pkl")):
            with open(os.path.join(self.pickle_dir, "q_seqs.pkl"), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "r_seqs.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "q_list.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "u_list.pkl"), "rb") as f:
                self.u_list = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "r_list.pkl"), "rb") as f:
                self.r_list = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "q2idx.pkl"), "rb") as f:
                self.q2idx = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "u2idx.pkl"), "rb") as f:
                self.u2idx = pickle.load(f)
        else:
            self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.r_list, self.q2idx, \
                self.u2idx = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        self.num_r = self.r_list.shape[0]

        self.q_seqs, self.r_seqs = \
            self.match_seq_len(self.q_seqs, self.r_seqs, max_seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_dir)
        df = df[(df["answered_correctly"] == 0) | (df["answered_correctly"] == 1)]

        u_list = np.unique(df["user_id"].values)
        q_list = np.unique(df["content_id"].values)
        r_list = np.unique(df["answered_correctly"].values)

        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}

        q_seqs = []
        r_seqs = []

        for u in u_list:
            df_u = df[df["user_id"] == u].sort_values("timestamp")

            q_seq = np.array([q2idx[q] for q in df_u["content_id"].values])
            r_seq = df_u["answered_correctly"].values

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)

        with open(os.path.join(self.pickle_dir, "q_seqs.pkl"), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.pickle_dir, "r_seqs.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.pickle_dir, "q_list.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.pickle_dir, "u_list.pkl"), "wb") as f:
            pickle.dump(u_list, f)
        with open(os.path.join(self.pickle_dir, "r_list.pkl"), "wb") as f:
            pickle.dump(r_list, f)
        with open(os.path.join(self.pickle_dir, "q2idx.pkl"), "wb") as f:
            pickle.dump(q2idx, f)
        with open(os.path.join(self.pickle_dir, "u2idx.pkl"), "wb") as f:
            pickle.dump(u2idx, f)

        return q_seqs, r_seqs, q_list, u_list, r_list, q2idx, u2idx

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