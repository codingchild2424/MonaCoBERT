
python train.py \
--model_fn normal_model.pth \
--model_name monacobert \
--dataset_name assist2009_pid \
--batch_size 256 \
--grad_acc True \
--grad_acc_iter 2 \
--n_epochs 1000