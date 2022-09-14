# MonaCoBERT: Monotonic attention based ConvBERT for Knowledge Tracing

This repository is for the research Monotonic attention based ConvBERT for Knowledge Tracing (https://arxiv.org/abs/2208.12615).  

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

- Batch size: Batch size was 512. You can use grad accumulation option, if you don't have enough GPU resources.
- Early stop: Early stop was 10.
- Training, validation, test ratio: Training ratio was 80%, test ratio was 20%, valid ratio was 10% of training ratio.
- Learning rate and optimizer: The learning rate was 0.001. Adam was used.


|Dataset | Metrics | DKT | DKVMN | SAKT | AKT | CL4KT | **MCB-NC** | **MCB -C**
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
|assist09 | AUC | 0.7285 | 0.7271 | 0.7179 | 0.7449 | 0.7600 | _0.8002_ | **0.8059**
| | RMSE | 0.4328 | 0.4348 | 0.4381 | 0.4413 | 0.4337 | **0.4029** | _0.4063_
|assist12 | AUC | 0.7006 | 0.7011 | 0.6998 | 0.7505 | 0.7314 | _0.8065_ | **0.8130**
| | RMSE | 0.4338 | 0.4355 | 0.4360 | 0.4250 | 0.4284 | _0.3976_ | **0.3935**
|assist17 | AUC | **0.7220** | 0.7095 | 0.6792 | 0.6803 | 0.6738 | 0.6700 | _0.7141_
| | RMSE | **0.4469** | _0.4516_ | 0.4591 | 0.4722 | 0.4713 | 0.4727 | 0.4630
|algebra05 | AUC | 0.8088 | 0.8146 | 0.8162 | 0.7673 | 0.7871 | _0.8190_ | **0.8201**
| | RMSE | 0.3703 | 0.3687 | _0.3685_ | 0.3918 | 0.3824 | 0.3940 | **0.3584**
|algebra06 | AUC | 0.7939 | 0.7961 | 0.7927 | 0.7505 | 0.7789 | _0.7997_ | **0.8064**
| | RMSE | _0.3666_ | **0.3661** | 0.3675 | 0.3986 | 0.3863 | 0.3835 | 0.3672
|EdNet | AUC | 0.6609 | 0.6602 | 0.6506 | 0.6687 | 0.6651 | _0.7221_ | **0.7336**
| | RMSE | 0.4598 | 0.4597 | 0.4629 | 0.4783 | 0.4750 | _0.4572_ | **0.4516**


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
   └── src
       ├── dataloaders
       ├── define_argparser.py
       ├── preprocess_data.py
       ├── get_modules
       ├── models
       ├── train.py
       ├── trainers
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
