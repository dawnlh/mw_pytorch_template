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
from srcs.tester._test_worker import test_worker

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
    parser.add_argument("--status",type=str,default=None)
    parser.add_argument("--checkpoint",type=str,default=None)
    args = parser.parse_args()
    return args

def main():
    ## parse arguments and cfgs
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.checkpoint is not None:
        cfg.tester.checkpoint = args.checkpoint
    if args.status is not None:
       cfg.runtime.status = args.status
    assert cfg.runtime.status!='train', "cfg.runtime.status shouldn't be set to 'train'"
    
    ## GPU setting
    if not cfg.runtime.gpus or cfg.runtime.gpus == -1:
        cfg.runtime.gpus = list(range(torch.cuda.device_count()))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cfg.runtime.gpus)) # set visible gpu ids
    cfg.runtime.n_gpus = len(cfg.runtime.gpus)
    cfg.runtime.distributed = cfg.runtime.n_gpus > 1

    ## make dirs
    datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    cfg.runtime.work_dir = osp.join(cfg.runtime.work_dir, cfg.runtime.exp_name, cfg.runtime.status, datetime)
    cfg.runtime.output_dir = osp.join(cfg.runtime.work_dir,"output")
    if not osp.exists(cfg.runtime.output_dir):
        os.makedirs(cfg.runtime.output_dir)


    ## setup logger, writer, tester
    logger = Logger(name='tester', log_path=osp.join(cfg.runtime.work_dir,'log.txt'))
    tester_name = 'srcs.tester.%s' % cfg.tester.name
    training_module = import_module(tester_name)
    Tester = training_module.Tester

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
    test_worker(Tester, cfg)


if __name__ == '__main__':
    main()


