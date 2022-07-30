import argparse
import torch

def define_argparser():
    p = argparse.ArgumentParser()

    # model_file_name
    p.add_argument('--model_fn', required=True)

    # basic arguments
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--train_ratio', type=float, default=.8)
    p.add_argument('--valid_ratio', type=float, default=.1)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=100)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--learning_rate', type=int, default = 0.001)

    # model, opt, dataset, crit arguments
    p.add_argument('--model_name', type=str, default='monacobert')
    p.add_argument('--optimizer', type=str, default='adam')
    p.add_argument('--dataset_name', type=str, default = 'assist2015')
    p.add_argument('--crit', type=str, default = 'binary_cross_entropy')

    # bidkt's arguments
    p.add_argument('--max_seq_len', type=int, default=100)
    p.add_argument('--num_encoder', type=int, default=12)
    p.add_argument('--hidden_size', type=int, default=512)
    p.add_argument('--num_head', type=int, default=16) # it will be divided 2(default) in attention class, so actual head num is 8(default)
    p.add_argument('--output_size', type=int, default=1) # KT is binary classification
    p.add_argument('--dropout_p', type=int, default=.1)
    p.add_argument('--use_leakyrelu', type=bool, default=True)

    # bigberd4kt's arguments
    p.add_argument('--num_random_blocks', type=int, default=3) # num_random_blocks = 3 
    p.add_argument('--block_size', type=int, default=5) # block_size = 64

    # grad_accumulation
    p.add_argument('--grad_acc', type=bool, default=False)
    p.add_argument('--grad_acc_iter', type=int, default=2)

    #five_fold cross validation
    p.add_argument('--fivefold', type=bool, default=False)

    config = p.parse_args()

    return config