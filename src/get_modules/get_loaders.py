from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from utils import collate_fn, pid_collate_fn, pid_time_collate_fn, pid_diff_collate_fn, pid_diff_pt_collate_fn
# assist2009
from dataloaders.assist2009_loader import ASSIST2009
from dataloaders.assist2009_pid_loader import ASSIST2009_PID
from dataloaders.assist2009_pid_diff_loader import ASSIST2009_PID_DIFF
# assist2012
from dataloaders.assist2012_loader import ASSIST2012
from dataloaders.assist2012_pid_loader import ASSIST2012_PID
from dataloaders.assist2012_pid_diff_loader import ASSIST2012_PID_DIFF
# assist2017
from dataloaders.assist2017_loader import ASSIST2017
from dataloaders.assist2017_pid_loader import ASSIST2017_PID
from dataloaders.assist2017_pid_diff_loader import ASSIST2017_PID_DIFF
# algebra2005
from dataloaders.algebra2005_loader import ALGEBRA2005
from dataloaders.algebra2005_pid_loader import ALGEBRA2005_PID
from dataloaders.algebra2005_pid_diff_loader import ALGEBRA2005_PID_DIFF
# algebra2006
from dataloaders.algebra2006_loader import ALGEBRA2006
from dataloaders.algebra2006_pid_loader import ALGEBRA2006_PID
from dataloaders.algebra2006_pid_diff_loader import ALGEBRA2006_PID_DIFF
# ednet
from dataloaders.ednet_loader import EDNET
from dataloaders.ednet_pid_loader import EDNET_PID
from dataloaders.ednet_pid_diff_loader import EDNET_PID_DIFF

def get_loaders(config, idx=None):

    # 1. choose the loaders

    # normal loaders
    if config.dataset_name == "assist2009":
        dataset = ASSIST2009(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = None
        num_diff = None
        collate = collate_fn
    elif config.dataset_name == "assist2012":
        dataset = ASSIST2012(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = None
        num_diff = None
        collate = collate_fn
    elif config.dataset_name == "assist2017":
        dataset = ASSIST2017(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = None
        num_diff = None
        collate = collate_fn
    elif config.dataset_name == "algebra2005":
        dataset = ALGEBRA2005(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = None
        num_diff = None
        collate = collate_fn
    elif config.dataset_name == "algebra2006":
        dataset = ALGEBRA2006(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = None
        num_diff = None
        collate = collate_fn
    elif config.dataset_name == "ednet":
        dataset = EDNET(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = None
        num_diff = None
        collate = collate_fn

    # pid loaders
    elif config.dataset_name == "assist2009_pid":
        dataset = ASSIST2009_PID(config.max_seq_len)  
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = None
        collate = pid_collate_fn
    elif config.dataset_name == "assist2012_pid":
        dataset = ASSIST2012_PID(config.max_seq_len)  
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = None
        collate = pid_collate_fn
    elif config.dataset_name == "assist2017_pid":
        dataset = ASSIST2017_PID(config.max_seq_len)  
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = None
        collate = pid_collate_fn
    elif config.dataset_name == "algebra2005_pid":
        dataset = ALGEBRA2005_PID(config.max_seq_len)  
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = None
        collate = pid_collate_fn
    elif config.dataset_name == "algebra2006_pid":
        dataset = ALGEBRA2006_PID(config.max_seq_len)  
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = None
        collate = pid_collate_fn
    elif config.dataset_name == "ednet_pid":
        dataset = EDNET_PID(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = None
        collate = pid_collate_fn

    # diff loader
    elif config.dataset_name == "assist2009_pid_diff":
        dataset = ASSIST2009_PID_DIFF(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = dataset.num_diff
        collate = pid_diff_collate_fn
    elif config.dataset_name == "assist2012_pid_diff":
        dataset = ASSIST2012_PID_DIFF(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = dataset.num_diff
        collate = pid_diff_collate_fn
    elif config.dataset_name == "assist2017_pid_diff":
        dataset = ASSIST2017_PID_DIFF(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = dataset.num_diff
        collate = pid_diff_collate_fn
    elif config.dataset_name == "algebra2005_pid_diff":
        dataset = ALGEBRA2005_PID_DIFF(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = dataset.num_diff
        collate = pid_diff_collate_fn
    elif config.dataset_name == "algebra2006_pid_diff":
        dataset = ALGEBRA2006_PID_DIFF(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = dataset.num_diff
        collate = pid_diff_collate_fn
    elif config.dataset_name == "ednet_pid_diff":
        dataset = EDNET_PID_DIFF(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = dataset.num_diff
        collate = pid_diff_collate_fn
    else:
        print("Wrong dataset_name was used...")

    # 2. data chunk

    # if fivefold = True
    if config.fivefold == True:

        first_chunk = Subset(dataset, range( int(len(dataset) * 0.2) ))
        second_chunk = Subset(dataset, range( int(len(dataset) * 0.2), int(len(dataset)* 0.4) ))
        third_chunk = Subset(dataset, range( int(len(dataset) * 0.4), int(len(dataset) * 0.6) ))
        fourth_chunk = Subset(dataset, range( int(len(dataset) * 0.6), int(len(dataset) * 0.8) ))
        fifth_chunk = Subset(dataset, range( int(len(dataset) * 0.8), int(len(dataset)) ))

        # idx from main
        # fivefold first
        if idx == 0:
            # train_dataset is 0.8 of whole dataset
            train_dataset = ConcatDataset([second_chunk, third_chunk, fourth_chunk, fifth_chunk])
            # valid_size is 0.1 of train_dataset
            valid_size = int( len(train_dataset) * config.valid_ratio)
            # train_size is 0.9 of train_dataset
            train_size = int( len(train_dataset) ) - valid_size
            
            train_dataset, valid_dataset = random_split(
                train_dataset, [ train_size, valid_size ]
            )
            # test_dataset is 0.2 of whole dataset
            test_dataset = first_chunk
        # fivefold second
        elif idx == 1:
            train_dataset = ConcatDataset([first_chunk, third_chunk, fourth_chunk, fifth_chunk])
            valid_size = int( len(train_dataset) * config.valid_ratio) #train의 0.1
            train_size = int( len(train_dataset) ) - valid_size #train의 0.9
            
            train_dataset, valid_dataset = random_split(
                train_dataset, [ train_size, valid_size ]
            )
            test_dataset = second_chunk
        # fivefold third
        elif idx == 2:
            train_dataset = ConcatDataset([first_chunk, second_chunk, fourth_chunk, fifth_chunk])
            valid_size = int( len(train_dataset) * config.valid_ratio) #train의 0.1
            train_size = int( len(train_dataset) ) - valid_size #train의 0.9
            
            train_dataset, valid_dataset = random_split(
                train_dataset, [ train_size, valid_size ]
            )
            test_dataset = third_chunk
        # fivefold fourth
        elif idx == 3:
            train_dataset = ConcatDataset([first_chunk, second_chunk, third_chunk, fifth_chunk])
            valid_size = int( len(train_dataset) * config.valid_ratio) #train의 0.1
            train_size = int( len(train_dataset) ) - valid_size #train의 0.9
            
            train_dataset, valid_dataset = random_split(
                train_dataset, [ train_size, valid_size ]
            )
            test_dataset = fourth_chunk
        # fivefold fifth
        elif idx == 4:
            train_dataset = ConcatDataset([first_chunk, second_chunk, third_chunk, fourth_chunk])
            valid_size = int( len(train_dataset) * config.valid_ratio) #train의 0.1
            train_size = int( len(train_dataset) ) - valid_size #train의 0.9
            
            train_dataset, valid_dataset = random_split(
                train_dataset, [ train_size, valid_size ]
            )
            test_dataset = fifth_chunk
    # fivefold = False
    else:
        train_size = int( len(dataset) * config.train_ratio * (1 - config.valid_ratio))
        valid_size = int( len(dataset) * config.train_ratio * config.valid_ratio)
        test_size = len(dataset) - (train_size + valid_size)

        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, [ train_size, valid_size, test_size ]
            )

    # 3. get DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size = config.batch_size,
        shuffle = True, # train_loader use shuffle
        collate_fn = collate
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size = config.batch_size,
        shuffle = False, # valid_loader don't use shuffle
        collate_fn = collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = config.batch_size,
        shuffle = False, # test_loader don't use shuffle
        collate_fn = collate
    )

    return train_loader, valid_loader, test_loader, num_q, num_r, num_pid, num_diff