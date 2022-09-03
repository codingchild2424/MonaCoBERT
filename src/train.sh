

datasets="assist2009_pid_diff algebra2005_pid_diff algebra2006_pid_diff assist2017_pid_diff ednet_pid_diff assist2012_pid_diff"

models="monabert_ctt cobert_ctt"

for model in ${models}
do
    python train.py \
    --model_fn ${model}_assist2009_pid_diff.pth \
    --model_name ${model} \
    --dataset_name assist2009_pid_diff \
    --fivefold True \
    --batch_size 128 \
    --grad_acc True \
    --grad_acc_iter 4 \
    --n_epochs 1000
done
