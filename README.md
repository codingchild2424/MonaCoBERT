# MonaCoBERT: Monotonic attention based ConvBERT for Knowledge Tracing

This repository is for the research Monotonic attention based ConvBERT for Knowledge Tracing (https://arxiv.org/abs/2208.12615).
(We changed the paper, and edited performance, it will be updated.)

If you find this code useful in your research, please cite
```
@misc{2208.12615,
   Author = {Unggi Lee and Yonghyun Park and Yujin Kim and Seongyune Choi and Hyeoncheol Kim},
   Title = {MonaCoBERT: Monotonic attention based ConvBERT for Knowledge Tracing},
   Year = {2022},
   Eprint = {arXiv:2208.12615},
}
```

# Performance (changed)

Dataset | Metrics | DKT | DKVMN | SAKT | AKT | CL4KT | MCB-NC | MCB -C
 ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
assist09 | AUC | 0.7285 | 0.7271 | 0.7179 | 0.7449 | 0.7600 | 0.8002 | 0.8059
^^ | RMSE | 0.4328 | 0.4348 | 0.4381 | 0.4413 | 0.4337 | 0.4029 | 0.4063


# Setups

1. We used docker environments, **ufoym/deefo**.  
   [https://hub.docker.com/r/ufoym/deepo/](https://hub.docker.com/r/ufoym/deepo/)
2. If you don't use docker environments, then you can use **requirements.txt**.

   ```
   pip install -r requirements.txt
   ```
3. You need to make directories for running code. However, some directory was not uploaded because of **.gitignore.** You can refer to D**irectory toggle** for making directories.

   <details><summary>Directory</summary>

   ```
   ├── README.md
   ├── checkpoints
   │   └── checkpoint.pt
   ├── datasets
   │   ├── algebra05
   │   │   └── preprocessed_df.csv
   │   ├── assistments09
   │   │   └── preprocessed_df.csv
   │   ├── assistments12
   │   │   └── preprocessed_df.csv
   │   ├── assistments17
   │   │   └── preprocessed_df.csv
   │   ├── bridge_algebra06
   │   │   └── preprocessed_df.csv
   │   └── ednet
   │       └── preprocessed_df.csv
   ├── model_records
   ├── requirements.txt
   ├── score_records
   │   └── auc_record.csv
   └── src
       ├── __pycache__
       │   ├── define_argparser.cpython-38.pyc
       │   └── utils.cpython-38.pyc
       ├── dataloaders
       │   ├── algebra2005_loader.py
       │   ├── algebra2005_pid_diff_loader.py
       │   ├── algebra2005_pid_loader.py
       │   ├── algebra2006_loader.py
       │   ├── algebra2006_pid_diff_loader.py
       │   ├── algebra2006_pid_loader.py
       │   ├── assist2009_loader.py
       │   ├── assist2009_pid_diff_loader.py
       │   ├── assist2009_pid_loader.py
       │   ├── assist2012_loader.py
       │   ├── assist2012_pid_diff_loader.py
       │   ├── assist2012_pid_loader.py
       │   ├── assist2017_loader.py
       │   ├── assist2017_pid_diff_loader.py
       │   ├── assist2017_pid_loader.py
       │   ├── ednet_loader.py
       │   ├── ednet_pid_diff_loader.py
       │   └── ednet_pid_loader.py
       ├── define_argparser.py
       ├── preprocess_data.py
       ├── get_modules
       │   ├── get_loaders.py
       │   ├── get_models.py
       │   └── get_trainers.py
       ├── models
       │   ├── monacobert.py
       │   └── monacobert_ctt.py
       ├── train.py
       ├── trainers
       │   ├── monacobert_ctt_trainer.py
       │   └── monacobert_trainer.py
       └── utils.py
   ```

   </details>
4. You can download the preprocessed dataset from our Google Drive.
   [https://drive.google.com/drive/folders/1B5vHabKUwzGEQAyDaj_0cBT6UqTMA13L?usp=sharing](https://drive.google.com/drive/folders/1B5vHabKUwzGEQAyDaj_0cBT6UqTMA13L?usp=sharing)
5. If you want to preprocess yourself, you can use **preprocess_data.py**.

   ```
   python preprocess_data.py --data_name assist09 --min_user_inter_num 5
   ```

# How to run this code?

If you want to run the MonaCoBERT, you have to use pid_loaders. For example,

```
python train.py --model_fn model.pth --model_name monacobert --dataset_name assist2009_pid
```

If you want to run the MonaCoBERT_CTT, you have to use pid_diff_loaders. For example,

```
python train.py --model_fn model.pth --model_name monacobert_ctt --dataset_name assist2009_pid_diff
```

If you want to use more options such as fivefold, gradient accumulation, you can refer to **define_argparser.py** and use like this.

```
python train.py --model_fn model.pth --model_name monacobert_ctt --dataset_name assist2009_pid_diff --fivefold True --grad_acc True --grad_acc_iter 2 
```


# Errata

If you have any question or find error in the code, you can send me a mail.

Contact: Unggi Lee ([codingchild@korea.ac.kr](mailto:codingchild@korea.ac.kr)).
