from trainers.monacobert_trainer import MonaCoBERT_Trainer
from trainers.monacobert_ctt_trainer import MonaCoBERT_CTT_Trainer

def get_trainers(model, optimizer, device, num_q, crit, config):

    # choose trainer
    if config.model_name == "monacobert":
        trainer = MonaCoBERT_Trainer(
            model=model,
            optimizer=optimizer,
            n_epochs=config.n_epochs,
            device=device,
            num_q=num_q,
            crit=crit,
            max_seq_len=config.max_seq_len,
            grad_acc=config.grad_acc,
            grad_acc_iter=config.grad_acc_iter
        )
    elif config.model_name == "monacobert_ctt":
        trainer = MonaCoBERT_CTT_Trainer(
            model=model,
            optimizer=optimizer,
            n_epochs=config.n_epochs,
            device=device,
            num_q=num_q,
            crit=crit,
            max_seq_len=config.max_seq_len,
            grad_acc=config.grad_acc,
            grad_acc_iter=config.grad_acc_iter
        )

    elif config.model_name == "bert_ctt":
        trainer = MonaCoBERT_CTT_Trainer(
            model=model,
            optimizer=optimizer,
            n_epochs=config.n_epochs,
            device=device,
            num_q=num_q,
            crit=crit,
            max_seq_len=config.max_seq_len,
            grad_acc=config.grad_acc,
            grad_acc_iter=config.grad_acc_iter
        )
    elif config.model_name == "monabert_ctt":
        trainer = MonaCoBERT_CTT_Trainer(
            model=model,
            optimizer=optimizer,
            n_epochs=config.n_epochs,
            device=device,
            num_q=num_q,
            crit=crit,
            max_seq_len=config.max_seq_len,
            grad_acc=config.grad_acc,
            grad_acc_iter=config.grad_acc_iter
        )
    elif config.model_name == "cobert_ctt":
        trainer = MonaCoBERT_CTT_Trainer(
            model=model,
            optimizer=optimizer,
            n_epochs=config.n_epochs,
            device=device,
            num_q=num_q,
            crit=crit,
            max_seq_len=config.max_seq_len,
            grad_acc=config.grad_acc,
            grad_acc_iter=config.grad_acc_iter
        )
    else:
        print("wrong trainer was choosed..")

    return trainer
