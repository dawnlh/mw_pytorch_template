from srcs.utils.registry import Registry,build_from_cfg 
import inspect
import torch

LOSSES = Registry("losses")

def register_torch_losses():
    torch_losses = []
    for name,obj in inspect.getmembers(torch.nn.modules.loss):
        if name.startswith('_'):
            continue
        if inspect.isclass(obj):
            LOSSES.register_module(obj)
            torch_losses.append(obj)
    return torch_losses

register_torch_losses()

def build_loss(cfg,default_args=None):
    losses = build_from_cfg(cfg, LOSSES, default_args)
    return losses