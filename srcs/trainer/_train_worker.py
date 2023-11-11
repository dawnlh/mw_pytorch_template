import platform
import torch
import torch.distributed as dist
from srcs.models.builder import build_model
from srcs.losses.builder import build_loss
from srcs.optimizers.builder import build_optimizer, build_scheduler
from srcs.datasets.builder import build_dataloader

# --------------------
# training logic
# --------------------

## training engine
def train_engine(Trainer,  cfg):
    if cfg.runtime.distributed:
        # multiple GPUs
        torch.multiprocessing.spawn(
            train_worker_ddp, nprocs=cfg.runtime.n_gpus, args=(Trainer,cfg))
    else:
        # single gpu
        train_worker(Trainer, cfg)

## training worker for single GPU
def train_worker(Trainer, cfg, local_rank=0):
    cfg.trainer.local_rank = local_rank
    # build dataloader
    train_dataloader, val_dataloader = build_dataloader(cfg)

    # build model
    model = build_model(cfg.model)
        
    # build optimizer & scheduler
    optimizer = build_optimizer(cfg.optimizer,{"params":model.parameters()})
    lr_scheduler = build_scheduler(cfg.lr_scheduler, {'optimizer':optimizer})

    # build loss
    criterion = build_loss(cfg.loss)

    # build metrics
    # `import` should be here, or os.environ["CUDA_VISIBLE_DEVICES"] will not work
    from srcs.metrics.builder import build_metric
    metrics = build_metric(cfg.metric)

    trainer = Trainer(model, optimizer, criterion, metrics, lr_scheduler, cfg,  train_dataloader, val_dataloader)
    trainer.train()

## training worker for multiple GPU
def train_worker_ddp(rank, Trainer, cfg):
    """
    Training with multiple GPUs
    """
    if(platform.system() == 'Windows'):
        backend = 'gloo'
    elif(platform.system() == 'Linux'):
        backend = 'nccl'
    else:
        raise RuntimeError('Unknown Platform (Windows and Linux are supported')
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:34567',
        world_size=cfg.runtime.n_gpus,
        rank=rank)
    
    # start training processes
    torch.cuda.set_device(rank)
    train_worker(Trainer, cfg, rank)
