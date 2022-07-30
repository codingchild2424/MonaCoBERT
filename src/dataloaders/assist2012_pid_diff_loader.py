import numpy as np
import pandas as pd

from torch.utils.data import Dataset

DATASET_DIR = "../datasets/assistments12/preprocessed_df.csv"

class ASSIST2012_PID_DIFF(Dataset):
    def __init__(self, max_seq_len, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        
        self.q_seqs, self.r_seqs, self.q_list, self.u_list, \
            self.r_list, self.q2idx, self.u2idx, self.pid_seqs, \
                self.diff_seqs, self.pid_list, self.diff_list = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        self.num_r = self.r_list.shape[0]
        self.num_pid = self.pid_list.shape[0]
        self.num_diff = self.diff_list.shape[0]

        self.q_seqs, self.r_seqs, self.pid_seqs, self.diff_seqs = \
            self.match_seq_len(self.q_seqs, self.r_seqs, self.pid_seqs, self.diff_seqs, max_seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index], self.pid_seqs[index], self.diff_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_dir, encoding="ISO-8859-1", sep='\t')
        df = df[(df["correct"] == 0) | (df["correct"] == 1)]

        u_list = np.unique(df["user_id"].values)
        q_list = np.unique(df["skill_id"].values)
        r_list = np.unique(df["correct"].values)
        pid_list = np.unique(df["item_id"].values)

        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}
        pid2idx = {pid: idx for idx, pid in enumerate(pid_list)} 

        # difficult
        diff = np.round(df.groupby('item_id')['correct'].mean() * 100)
        diff_list = np.unique(diff)

        q_seqs = []
        r_seqs = []
        pid_seqs = []
        diff_seqs = []

        for u in u_list:
            df_u = df[df["user_id"] == u]

            q_seq = np.array([q2idx[q] for q in df_u["skill_id"].values])
            r_seq = df_u["correct"].values
            pid_seq = np.array([pid2idx[pid] for pid in df_u["item_id"].values])
            diff_seq = np.array([diff[item] for item in df_u["item_id"].values])

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
            pid_seqs.append(pid_seq)
            diff_seqs.append(diff_seq)

        return q_seqs, r_seqs, q_list, u_list, r_list, q2idx, u2idx, pid_seqs, diff_seqs, pid_list, diff_list #끝에 두개 추가

    def match_seq_len(self, q_seqs, r_seqs, pid_seqs, diff_seqs, max_seq_len, pad_val=-1):

        proc_q_seqs = []
        proc_r_seqs = []
        proc_pid_seqs = []
        proc_diff_seqs = []

        for q_seq, r_seq, pid_seq, diff_seq in zip(q_seqs, r_seqs, pid_seqs, diff_seqs):

            i = 0
            while i + max_seq_len < len(q_seq):
                proc_q_seqs.append(q_seq[i:i + max_seq_len - 1])
                proc_r_seqs.append(r_seq[i:i + max_seq_len - 1])
                proc_pid_seqs.append(pid_seq[i:i + max_seq_len - 1])
                proc_diff_seqs.append(diff_seq[i:i + max_seq_len - 1])

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
            proc_pid_seqs.append(
                np.concatenate(
                    [
                        pid_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
            proc_diff_seqs.append(
                np.concatenate(
                    [
                        diff_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )

        return proc_q_seqs, proc_r_seqs, proc_pid_seqs, proc_diff_seqs