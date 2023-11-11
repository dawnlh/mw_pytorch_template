import os
import os.path as osp
from srcs.utils.config import Config
from srcs.utils.logger import Logger
from srcs.utils.utils import get_device_info
from importlib import import_module
import torch
import numpy as np
import time
import argparse 
import json 
from srcs.trainer._train_worker import train_engine

# fix random seeds for reproducibility
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# --------------------
# main
# --------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=str)
    parser.add_argument("--status",type=str,default='train')
    parser.add_argument("--resume",type=str,default=None)
    args = parser.parse_args()
    return args

def main():
    ## parse arguments and cfgs
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.status is not None:
        cfg.runtime.status = args.status
    if args.resume is not None:
        cfg.trainer.resume = args.resume
    # assert cfg.runtime.status=='train', "please set cfg.runtime.status to 'train'"

    ## GPU setting
    if not cfg.runtime.gpus or cfg.runtime.gpus == -1:
        cfg.runtime.gpus = list(range(torch.cuda.device_count()))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cfg.runtime.gpus)) # set visible gpu ids
    cfg.runtime.n_gpus = len(cfg.runtime.gpus)
    cfg.runtime.distributed = cfg.runtime.n_gpus > 1

    ## make dirs
    datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    cfg.runtime.work_dir = osp.join(cfg.runtime.work_dir, cfg.runtime.exp_name, cfg.runtime.status, datetime)
    cfg.runtime.tb_dir = osp.join(cfg.runtime.work_dir,"tensorboard")
    cfg.runtime.output_dir = osp.join(cfg.runtime.work_dir,"output")
    cfg.runtime.checkpoints_dir = osp.join(cfg.runtime.work_dir,"checkpoints")

    if not osp.exists(cfg.runtime.tb_dir):
        os.makedirs(cfg.runtime.tb_dir)
    if not osp.exists(cfg.runtime.output_dir):
        os.makedirs(cfg.runtime.output_dir)
    if not osp.exists(cfg.runtime.checkpoints_dir):
        os.makedirs(cfg.runtime.checkpoints_dir)

    ## setup logger, writer, trainer
    logger = Logger(name='trainer', log_path=osp.join(cfg.runtime.work_dir,'log.txt'))
    trainer_name = 'srcs.trainer.%s' % cfg.trainer.name
    training_module = import_module(trainer_name)
    Trainer = training_module.Trainer

    ## log env & cfg info
    dash_line = '=' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k,v) for k, v in device_info.items()])
    logger.info('GPU info:\n' 
            + dash_line + 
            env_info + '\n' +
            dash_line)
    Config.tofile(cfg, osp.join(cfg.runtime.work_dir, 'config.json'))
    logger.info('Config info:\n'
            + dash_line + 
            json.dumps(cfg, indent=4)+'\n'+
            dash_line) 
    
    ## conduct training
    train_engine(Trainer, cfg)


if __name__ == '__main__':
    main()


