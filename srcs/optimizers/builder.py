from srcs.utils.registry import Registry,build_from_cfg 
import inspect
import torch

## Registry
OPTIMIZERS = Registry('optimizer')
SCHEDULERS = Registry('scheduler')

## register
def register_torch_optimizers():
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            OPTIMIZERS.register_module(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers

def register_torch_schedulers():
    torch_schedulers = []
    for module_name in dir(torch.optim.lr_scheduler):
        if module_name.startswith('__'):
            continue
        _sched = getattr(torch.optim.lr_scheduler, module_name)
        if inspect.isclass(_sched) and issubclass(_sched,
                                                  torch.optim.lr_scheduler._LRScheduler):
            SCHEDULERS.register_module(_sched)
            torch_schedulers.append(module_name)
    return torch_schedulers


## build
def build_scheduler(cfg,default_args=None):
    scheduler = build_from_cfg(cfg, SCHEDULERS, default_args)
    return scheduler
def build_optimizer(cfg,default_args=None):
    optimizer = build_from_cfg(cfg, OPTIMIZERS, default_args)
    return optimizer


## do register
register_torch_optimizers()
register_torch_schedulers()