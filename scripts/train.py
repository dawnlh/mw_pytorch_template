import os
import os.path as osp
from shutil import copyfile
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

from srcs.datasets.builder import build_dataloader 
from srcs.models.builder import build_model
from srcs.optimizers.builder import  build_optimizer,build_scheduler
from srcs.losses.builder import build_loss
from srcs.metrics.builder import build_metric
from srcs.utils.config import Config
from srcs.utils.logger import Logger
from srcs.utils.utils import load_checkpoints, get_device_info, collect
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import time
import argparse 
import json 
import einops

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=str)
    parser.add_argument("--work_dir",type=str,default=None)
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--distributed",type=bool,default=False)
    parser.add_argument("--resume",type=str,default=None)
    parser.add_argument("--local_rank",default=-1)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    local_rank = int(args.local_rank) 
    if args.distributed:
        args.device = torch.device("cuda",local_rank)
    return args

def main():

    # ==================== init ========================
    ## arg parse
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir is not None:
        cfg.runtime.work_dir = args.work_dir
    if args.resume is not None:
        cfg.trainer.resume = args.resume
    cfg.runtime.distributed = args.distributed
    cfg.runtime.n_gpus = torch.cuda.device_count()
    cfg.runtime.status = 'train'
    
    work_dir = cfg.runtime.work_dir
    device = args.device
    is_distributed = args.distributed
    n_gpus = cfg.runtime.n_gpus


    ## dir setting
    datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    work_dir = osp.join(work_dir, cfg.runtime.exp_name, cfg.runtime.status, datetime)
    tb_dir = osp.join(work_dir,"tensorboard")
    output_dir = osp.join(work_dir,"output")
    checkpoints_dir = osp.join(work_dir,"checkpoints")

    if not osp.exists(tb_dir):
        os.makedirs(tb_dir)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    if not osp.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    ## logger & writer
    logger = Logger(name='trainer', log_path=osp.join(work_dir,'log.txt'))
    writer = SummaryWriter(log_dir = tb_dir)

    # ======= model, optimizer, loss, metrics, & dataloader ========
    ## model & ddp
    rank = 0 
    if is_distributed:
        local_rank = int(args.local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    model = build_model(cfg.model).to(device)
    
    ## optimizer
    optimizer = build_optimizer(cfg.optimizer,{"params":model.parameters()})
    
    ## scheduler
    lr_scheduler = build_scheduler(cfg.lr_scheduler, {'optimizer':optimizer})

    ## loss
    criterion = build_loss(cfg.loss)
    criterion = criterion.to(device)

    # build metrics
    metrics = build_metric(cfg.metric)
    
    ## dataloader
    train_dataloader, val_dataloader = build_dataloader(cfg)

    ## info log
    dash_line = '=' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k,v) for k, v in device_info.items()])
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    param_num = f'Trainable parameters: {sum([p.numel() for p in trainable_params])/1e6:.2f} M\n'
    if rank==0:
        Config.tofile(cfg, osp.join(work_dir, 'config.json'))
        logger.info('GPU info:\n' 
                + dash_line + 
                env_info + '\n' +
                dash_line)
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
                    f'train: {len(train_dataloader.dataset)}' +'\n' +
                    f'val: {len(val_dataloader.dataset)}' + '\n'+
                    dash_line)
        
    # ==================== train ========================
    ## resume checkpoint
    start_epoch = 0
    if rank==0:
        if cfg.trainer.resume is not None:
            logger.info(f"Load pre_train model from: {cfg.trainer.resume}")
            resume_dict = torch.load(cfg.trainer.resume)
            state_dict = resume_dict["state_dict"]
            load_checkpoints(model,state_dict)
            if 'epoch' in cfg.trainer.resume_conf:
                start_epoch = resume_dict["epoch"]
                logger.info(f"Resume from epoch {start_epoch}...")
            if 'optimizer' in cfg.trainer.resume_conf:
                optim_state_dict = resume_dict["optimizer"]
                optimizer.load_state_dict(optim_state_dict)
                logger.info(f"Resume optimizer")
        else:            
            logger.info("📣 No pre_train model\n")


    ## DDP init
    if is_distributed:
        model = DDP(model,
                    device_ids=[local_rank],
                    output_device=local_rank)
    ## train loop
    iter_num = len(train_dataloader) 
    clock_start = time.time()
    for epoch in range(start_epoch,cfg.trainer.num_epochs):
        # ------------------- train ----------------------
        epoch_loss = 0
        model = model.train()
        start_time = time.time()
        for iteration, (img_noise, img_target) in enumerate(train_dataloader):
            # data to device
            img_noise, img_target = img_noise.to(device), img_target.to(device)
            # forward
            model_out = model(img_noise)
            # loss
            loss = criterion(model_out, img_target)
            epoch_loss += loss.item()
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # info record
            metric_str = ''
            if iteration % cfg.trainer.logging_interval == 0:
                lr = optimizer.state_dict()["param_groups"][0]["lr"]
                iter_metrics = metrics(model_out, img_target)
                iter_metrics.update({"loss":loss})
                for k, v_ in iter_metrics.items():
                    v = collect(v_) if is_distributed else v
                    v = v.item() if isinstance(v, torch.Tensor) else v
                    writer.add_scalar(f'[train]/{k}', v, epoch*len(train_dataloader) + iteration)
                    metric_str += f'{k}: {v:.5f} '
                iter_len = len(str(iter_num))
                if rank==0:
                    logger.info("Train: [{}][{:>{}}/{}]\tlr: {:.6f}, loss: {:.5f} {}".format(epoch+1,iteration*n_gpus,iter_len,iter_num*n_gpus,lr,loss.item(),metric_str))
        end_time = time.time()
        
        # lr scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        if rank==0:
            # save checkpoint
            if is_distributed:
                save_model = model.module
            else:
                save_model = model
            arch = type(save_model).__name__
            state = {
                'arch': arch,
                "epoch": epoch+1, 
                "state_dict": save_model.state_dict(), 
                "optimizer": optimizer.state_dict(), 
                'config': cfg
            }
            cur_ckp = osp.join(checkpoints_dir,"ckp-epoch"+str(epoch+1)+".pth")
            torch.save(state,cur_ckp) 

            # remove old ckp
            save_latest_k = cfg.trainer.get('save_latest_k', False) 
            if save_latest_k > 0: 
                latest_path = osp.join(checkpoints_dir,'model_latest.pth')
                copyfile(cur_ckp, latest_path)
                outdated_path = osp.join(checkpoints_dir,"ckp-epoch"+str(epoch+1-save_latest_k)+".pth")
                try:
                    os.remove(outdated_path)
                except FileNotFoundError:
                    # this happens when current model is loaded from checkpoint
                    # or target file is already removed somehow
                    pass
        
        # ------------------- val ----------------------
        if val_dataloader and epoch % cfg.trainer.eval_interval==0:
            iter_metrics, demo_res = validate(model, val_dataloader, device, criterion, metrics, is_distributed)
            cur_time = time.time()
            if rank==0:
                # writer
                metric_str = ''
                for k, v in iter_metrics.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    writer.add_scalar(f'[valid]/{k}', v, epoch)
                    metric_str += f'{k}: {v:.5f} '
                
                for k, v in demo_res.items():
                    writer.add_image(f'[valid]/{k}', make_grid(demo_res[k][0:8, ...].cpu(), nrow=2, normalize=True), epoch)

                # logger
                logger.info('-'*80+f'\nValid: [{epoch+1}] epoch time: {(end_time-start_time)/60:.2f} min total time: {(cur_time-clock_start)/3600:.2f} h | {metric_str}\n'+'='*80+'\n')
            

            

def validate(model, val_dataloader, device, criterion, metrics, is_distributed):
    """
    Validate after training an epoch
    """
    model.eval()
    iter_metrics = {}
    with torch.no_grad():
        for batch_idx, (img_noise, img_target) in enumerate(val_dataloader):
            img_noise, img_target = img_noise.to(
                device), img_target.to(device)
            
            # forward
            output = model(img_noise)
            # loss calc
            loss = criterion(output, img_target)
            # metrics
            calc_metrics = metrics(output, img_target)
            calc_metrics.update({'loss':loss})
            
            # average metric between processes
            for k, v in calc_metrics.items():
                vv = collect(v) if is_distributed > 1 else v
                iter_metrics.update({k:iter_metrics.get(k, 0)+vv})

        # average iter metrics
        iter_metrics = {k:v/(batch_idx+1) for k,v in iter_metrics.items()}
        # iter images
        demo_res = {
            'input': img_noise[0:4, ...], 'img_target': img_target[0:4, ...], 'output': output[0:4, ...]}

        return iter_metrics, demo_res
    
    
if __name__ == '__main__':
    main()