from models.monacobert import MonaCoBERT
from models.monacobert_ctt import MonaCoBERT_CTT

from models.bert_ctt import BERT_CTT
from models.monabert_ctt import MonaBERT_CTT
from models.cobert_ctt import CoBERT_CTT

# get models
def get_models(num_q, num_r, num_pid, num_diff, device, config):
    # choose the models
    if config.model_name == "monacobert":
        model = MonaCoBERT(
            num_q=num_q,
            num_r=num_r,
            num_pid=num_pid,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_head=config.num_head,
            num_encoder=config.num_encoder,
            max_seq_len=config.max_seq_len,
            device=device,
            use_leakyrelu=config.use_leakyrelu,
            dropout_p=config.dropout_p
        ).to(device)
    elif config.model_name == "monacobert_ctt":
        model = MonaCoBERT_CTT(
            num_q=num_q,
            num_r=num_r,
            num_pid=num_pid,
            num_diff=num_diff,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_head=config.num_head,
            num_encoder=config.num_encoder,
            max_seq_len=config.max_seq_len,
            device=device,
            use_leakyrelu=config.use_leakyrelu,
            dropout_p=config.dropout_p
        ).to(device)
    elif config.model_name == "bert_ctt":
        model = BERT_CTT(
            num_q=num_q,
            num_r=num_r,
            num_pid=num_pid,
            num_diff=num_diff,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_head=config.num_head,
            num_encoder=config.num_encoder,
            max_seq_len=config.max_seq_len,
            device=device,
            use_leakyrelu=config.use_leakyrelu,
            dropout_p=config.dropout_p
        ).to(device)
    elif config.model_name == "monabert_ctt":
        model = MonaBERT_CTT(
            num_q=num_q,
            num_r=num_r,
            num_pid=num_pid,
            num_diff=num_diff,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_head=config.num_head,
            num_encoder=config.num_encoder,
            max_seq_len=config.max_seq_len,
            device=device,
            use_leakyrelu=config.use_leakyrelu,
            dropout_p=config.dropout_p
        ).to(device)
    elif config.model_name == "cobert_ctt":
        model = CoBERT_CTT(
            num_q=num_q,
            num_r=num_r,
            num_pid=num_pid,
            num_diff=num_diff,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_head=config.num_head,
            num_encoder=config.num_encoder,
            max_seq_len=config.max_seq_len,
            device=device,
            use_leakyrelu=config.use_leakyrelu,
            dropout_p=config.dropout_p
        ).to(device)
    else:
        print("Wrong model_name was used...")

    return model