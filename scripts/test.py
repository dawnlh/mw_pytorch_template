import json
import os
import os.path as osp
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch 
from srcs.metrics.builder import build_metric
from srcs.datasets.builder import build_dataloader 
from srcs.utils.utils import load_checkpoints
from srcs.utils.image import tensor2uint, imsave_n, imsave
from srcs.utils.config import Config
from srcs.models.builder import build_model
from srcs.utils.logger import Logger
import numpy as np 
import argparse 
import time
import einops 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=str)
    parser.add_argument("--work_dir",type=str)
    parser.add_argument("--checkpoint",type=str)
    parser.add_argument("--status",type=str,default="test")
    parser.add_argument("--save_img",type=bool,default=False)
    parser.add_argument("--device",type=str,default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device="cpu"
    return args

def main():

    # ==================== init ========================
    ## arg parse
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir is not None:
        cfg.runtime.work_dir = args.work_dir
    if args.checkpoint is not None:
        cfg.tester.checkpoint = args.checkpoint
    if args.status is not None:
        cfg.runtime.status = args.status
    if args.save_img is not None:
        cfg.tester.save_img = args.save_img
    work_dir = cfg.runtime.work_dir
    save_img = cfg.tester.save_img
    device = args.device


    ## dir setting
    datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    work_dir = osp.join(work_dir, cfg.runtime.exp_name, cfg.runtime.status, datetime)
    output_dir = osp.join(work_dir,"output")
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    if save_img:
        os.makedirs(output_dir+'/images')

    ## logger
    logger=Logger(name='tester', log_path=osp.join(work_dir,'log.txt'))


    # ======= model, metrics, & dataloader ========
    ## model 
    model = build_model(cfg.model).to(device)

    # build metrics
    metrics = build_metric(cfg.metric)

    ## build dataloader
    test_dataloader = build_dataloader(cfg)    

    ## info log
    dash_line = '=' * 80 + '\n'
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    param_num = f'Trainable parameters: {sum([p.numel() for p in trainable_params])/1e6:.2f} M\n'
    Config.tofile(cfg, osp.join(work_dir, 'config.json'))
    logger.info('Config info:\n'
            + dash_line + 
            json.dumps(cfg, indent=4)+'\n'+
            dash_line) 
    logger.info('Model info:\n'
            + dash_line + 
            str(model)+'\n'+ param_num +'\n'+
            dash_line)
    logger.info('Dataset info:\n'
                + dash_line + 
                f'num_samples: {len(test_dataloader.dataset)}' +'\n' +
                dash_line)

    # ==================== test ========================
    ## load checkpoint
    logger.info(f"Load pre_train model from: \n\t{cfg.tester.checkpoint}")
    resume_dict = torch.load(cfg.tester.checkpoint)
    state_dict = resume_dict["state_dict"]
    load_checkpoints(model,state_dict)

    ## test loop
    model.eval()
    total_metrics = {}
    time_start = time.time()
    with torch.no_grad():
        for batch_idx, (img_noise, img_target) in enumerate(test_dataloader):
            img_noise, img_target = img_noise.to(
                    device), img_target.to(device)
            # forward
            output = model(img_noise)

            # save image
            if save_img:
                for k, (in_img, out_img, gt_img) in enumerate(zip(img_noise, output, img_target)):
                    in_img = tensor2uint(in_img)
                    out_img = tensor2uint(out_img)
                    gt_img = tensor2uint(gt_img)
                    imgs = [in_img, out_img, gt_img]
                    imsave_n(imgs, f'{output_dir}/images/test{batch_idx+1:03d}_{k+1:03d}.png')
                    # imsave(in_img, f'{cfg.runtime.output_dir}/images/test{i+1:03d}_{k+1:03d}_in.png')
            
            if cfg.runtime.status != 'realexp':
                cur_batch_size = img_noise.shape[0]
                calc_metrics = metrics(output, img_target)
                for k, v in calc_metrics.items():
                    total_metrics.update({k:total_metrics.get(k,0) + v * cur_batch_size})

        # time cost
        time_end = time.time()
        time_cost = time_end-time_start
        n_samples = len(test_dataloader.sampler)
        
        # metrics average
        test_metrics = {k: v / n_samples for k, v in total_metrics.items()}
        metrics_str = ' '.join([f'{k}: {v:6.4f}' for k, v in test_metrics.items()])

        logger.info('='*80 + f'\n🎯 time/sample {time_cost/n_samples:6.4f} ' + metrics_str + '\n' + '='*80)

if __name__=="__main__":
    main()