# MonaCoBERT: Monotonic attention based ConvBERT for Knowledge Tracing

This repository is for the research Monotonic attention based ConvBERT for Knowledge Tracing (MonaCoBERT).


# Setups

You can use requirements.txt

```
pip install -r requirements.txt
```


# How to run this code?

If you want to run the MonaCoBERT, you have to use pid_loaders. For example,

```
python train.py --model_fn model.pth --model_name monacobert --dataset_name assist2009_pid
```

If you want to run the MonaCoBERT_CTT, you have to use ₩pid_diff_loaders₩. For example,

```
python train.py --model_fn model.pth --model_name monacobert_ctt --dataset_name assist2009_pid_diff
```

If you want to use more options such as fivefold, gradient accumulation, you can refer to define_argparser.py.
