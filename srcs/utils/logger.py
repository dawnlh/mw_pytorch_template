import os
import pandas as pd
from itertools import product
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging, logging.config
import os, time
import os.path as osp
from torchvision.utils import make_grid
import torch 

def Logger(name=None, log_path='./runtime.log'):
    config_dict = {
        "version": 1,
        "formatters": {
            "simple": {
            "format": "%(message)s"
            },
            "detailed": {
            "format": "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
            }
        },
        "handlers": {
            "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
            },
            "file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": log_path
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file"]
        },
        "disable_existing_loggers": False
    }
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)
    return logger 

class TensorboardWriter():
    def __init__(self, log_dir, enabled):
        self.writer = SummaryWriter(log_dir) if enabled else None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

        self.step = 0

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding', 'add_hparams'
        }
        self.timer = datetime.now()

    def set_step(self, step, speed_chk=None):  # phases = 'train'|'valid'|None
        self.step = step
        # measure the calculation speed by call this function between 2 steps (steps_per_sec)
        if speed_chk and step != 0:
            duration = datetime.now() - self.timer
            self.add_scalar(f'steps_per_sec/{speed_chk}',
                            1 / duration.total_seconds())
        self.timer = datetime.now()
    
    def writer_update(self, step, phase, metrics, image_tensors=None):
        # writer update
        self.set_step(step, speed_chk=f'{phase}')

        metric_str = ''
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.add_scalar(f'{phase}/{k}', v)
                metric_str += f'{k}: {v:8.5f} '
        
        if image_tensors:
            for k, v in image_tensors.items():
                self.add_image(
                    f'{phase}/{k}', make_grid(image_tensors[k][0:8, ...].cpu(), nrow=2, normalize=True))
        
        return metric_str # return metric string for logger
    

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(var1, var2, *args, **kwargs):
                if add_data is not None:
                    add_data(var1, var2, self.step, *args, **kwargs)
            return wrapper
        else:
            attr = getattr(self.writer, name, None)
            if not attr:
                raise AttributeError('unimplemented attribute')
            return attr
        
class BatchMetrics:
    def __init__(self, *keys, postfix='', writer=None):
        self.writer = writer
        self.postfix = postfix
        if postfix:
            keys = [k+postfix for k in keys]
        self._data = pd.DataFrame(
            index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.postfix:
            key = key + self.postfix
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / \
            self._data.counts[key]

    def avg(self, key):
        if self.postfix:
            key = key + self.postfix
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class EpochMetrics:
    def __init__(self, metric_names, phases=('train', 'valid'), monitoring='off', writer=None):
        self.logger = Logger('epoch-metrics')
        # setup pandas DataFrame with hierarchical columns
        columns = tuple(product(metric_names, phases))
        self._data = pd.DataFrame(columns=columns)
        self.monitor_mode, self.monitor_metric = self._parse_monitoring_mode(
            monitoring)
        self.topk_idx = []
        self.writer = writer

    def minimizing_metric(self, idx):
        if self.monitor_mode == 'off':
            return 0
        try:
            metric = self._data[self.monitor_metric].loc[idx]
        except KeyError:
            self.logger.warning("Warning: Metric '{}' is not found. "
                                "Model performance monitoring is disabled.".format(self.monitor_metric))
            self.monitor_mode = 'off'
            return 0
        if self.monitor_mode == 'min':
            return metric
        else:
            return - metric

    def _parse_monitoring_mode(self, monitor_mode):
        if monitor_mode == 'off':
            return 'off', None
        else:
            monitor_mode, monitor_metric = monitor_mode.split()
            monitor_metric = tuple(monitor_metric.split('/'))
            assert monitor_mode in ['min', 'max']
        return monitor_mode, monitor_metric

    def is_improved(self):
        if self.monitor_mode == 'off':
            return True

        last_epoch = self._data.index[-1]
        best_epoch = self.topk_idx[0]
        return last_epoch == best_epoch

    def keep_topk_checkpt(self, checkpt_dir, k=3):
        """
        Keep top-k checkpoints by deleting k+1'th best epoch index from dataframe for every epoch.
        """
        if len(self.topk_idx) > k and self.monitor_mode != 'off':
            last_epoch = self._data.index[-1]
            self.topk_idx = self.topk_idx[:(k+1)]
            if last_epoch not in self.topk_idx:
                to_delete = last_epoch
            else:
                to_delete = self.topk_idx[-1]

            # delete checkpoint having out-of topk metric
            filename = str(
                checkpt_dir / 'ckp-epoch{}.pth'.format(to_delete.split('-')[1]))
            try:
                os.remove(filename)
            except FileNotFoundError:
                # this happens when current model is loaded from checkpoint
                # or target file is already removed somehow
                pass

    def update(self, epoch, result):
        epoch_idx = f'epoch-{epoch}'
        self._data.loc[epoch_idx] = {
            tuple(k.split('/')): v for k, v in result.items()}

        self.topk_idx.append(epoch_idx)
        self.topk_idx = sorted(self.topk_idx, key=self.minimizing_metric)

        # write epoch info to tensorboard
        if self.writer is not None:
            for k, v in result.items():
                self.writer.add_scalar(k + '/epoch', v)

    def latest(self):
        return self._data[-1:]

    def to_csv(self, save_path=None):
        self._data.to_csv(save_path)

    def __str__(self):
        return str(self._data)
