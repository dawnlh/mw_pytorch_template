import torch
from srcs.utils.registry import Registry,build_from_cfg 
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

DATASETS = Registry("dataset")

def build_dataset(cfg,default_args=None):
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset

def build_dataloader(cfg):
    if cfg.runtime.status=='train':
        return build_train_dataloader(cfg)
    elif cfg.runtime.status in ['test', 'realexp', 'simuexp']:
        return build_test_dataloader(cfg)
    else:
        raise NotImplementedError(f"status {cfg.runtime.status} not implemented")


def build_train_dataloader(cfg):
    """
    Build dataloader for training and validation
    """
    train_dataset = build_dataset(cfg.train_dataloader.dataset)
    val_part = cfg.train_dataloader.val_part
    if isinstance(val_part, (int, float)):
        assert 0 <= val_part < 1, "val_split should be within 0 to 1"
        num_total = len(train_dataset)
        num_valid = int(num_total * val_part)
        num_train = num_total - num_valid
        train_dataset, val_dataset = random_split(train_dataset, [num_train, num_valid])
    elif isinstance(val_part, dict):
        val_dataset = build_dataset(val_part)
    else:
        val_dataset = []
    
    if not val_dataset:
        # val_dataset is empty
        val_dataloader = None

    if cfg.runtime.distributed:
        # multi GPU
        train_sampler = DistributedSampler(train_dataset,shuffle=True)
        train_dataloader = DataLoader(dataset=train_dataset, 
                                        batch_size=cfg.train_dataloader.batch_size,
                                        sampler=train_sampler,
                                        num_workers = cfg.train_dataloader.num_workers)
        if val_dataset:
            val_sampler = DistributedSampler(val_dataset,shuffle=True)
            val_dataloader = DataLoader(dataset=val_dataset, 
                                            batch_size=cfg.train_dataloader.batch_size,
                                            sampler=val_sampler,
                                            num_workers = cfg.train_dataloader.num_workers)
    else:
        train_dataloader = DataLoader(dataset=train_dataset, 
                                        batch_size=cfg.train_dataloader.batch_size,
                                        shuffle=True,
                                        num_workers = cfg.train_dataloader.num_workers)
        if val_dataset:
            val_dataloader = DataLoader(dataset=val_dataset, 
                                        batch_size=cfg.train_dataloader.batch_size,
                                        shuffle=True,
                                        num_workers = cfg.train_dataloader.num_workers)
    return train_dataloader, val_dataloader

def build_test_dataloader(cfg):
    """
    Build dataloader for testing and exp
    """
    test_dataset = build_dataset(cfg.test_dataloader.dataset)
    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=cfg.test_dataloader.batch_size,
                                  shuffle=False,
                                  num_workers=cfg.test_dataloader.num_workers)
    return test_dataloader