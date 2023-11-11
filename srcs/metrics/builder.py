from srcs.utils.registry import Registry,build_from_cfg 

METRICS = Registry("metrics")


def build_metric(cfg,default_args=None):
    metrics = build_from_cfg(cfg, METRICS, default_args)
    return metrics