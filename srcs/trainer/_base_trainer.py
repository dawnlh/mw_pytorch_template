# ======================================
# BaseTrainer for basic network
# ======================================
import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from abc import abstractmethod, ABCMeta
from torchvision.utils import make_grid
from shutil import copyfile
from datetime import datetime
from ._utils import is_master
from os.path import join as opj
from srcs.utils.logger import Logger, TensorboardWriter

class BaseTrainer(metaclass=ABCMeta):
    """
    Base class for all trainers
    """

    def __init__(self, model, optimizer, criterion, metrics, lr_scheduler, config,  train_dataloader, val_dataloader=None):

        ## param assignment
        self.config = config
        self.device = config.trainer.local_rank
        self.model = model.to(self.device)
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        cfg_trainer = config.trainer

        ## resume
        if config.trainer.resume is not None:
            resume_conf = cfg_trainer.get('resume_conf', None)
            if resume_conf is None:
                resume_conf = ['epoch', 'optimizer']
            self.resume_checkpoint(config.trainer.resume, resume_conf)
        
        ## logger, writer &metrics
        self.logger = Logger(name='trainer', log_path=opj(config.runtime.work_dir,'log.txt'))
        self.logging_interval = cfg_trainer.get('logging_interval', 100)
        if is_master():
            self.writer = TensorboardWriter(
                config.runtime.tb_dir, config.runtime.tensorboard)
        else:
            self.writer = TensorboardWriter(config.runtime.tb_dir, False)

        ## runtime
        self.start_epoch = 1
        self.num_epochs = cfg_trainer.get('num_epochs', int(1e10))
        if self.num_epochs is None:
            self.num_epochs = int(1e10)
        self.eval_interval = cfg_trainer.get('eval_interval', 1)
        self.save_latest_k = cfg_trainer.get('save_latest_k', -1)
        self.milestone_ckp = cfg_trainer.get('milestone_ckp', [])
        self.max_iter = cfg_trainer.get('max_iter', None)
        self.max_iter = self.max_iter if self.max_iter else len(train_dataloader)

        ## model DDP
        if config.runtime.n_gpus > 1:
            # multi GPU DDP
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[self.device], output_device=self.device)

        ## info log
        if is_master():
            # log model info
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            param_num = f'Trainable parameters: {sum([p.numel() for p in trainable_params])/1e6:.2f} M\n'
            dash_line = '=' * 80 + '\n'
            self.logger.info('Model info:\n'
                        + dash_line + 
                        str(model)+'\n'+ param_num +'\n'+
                        dash_line)
            self.logger.info('Dataset info:\n'
                        + dash_line + 
                        f'train: {len(train_dataloader.dataset)}' +'\n' +
                        f'val: {len(val_dataloader.dataset)}' + '\n'+
                        dash_line)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        """
        Validation logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError


    def train(self):
        """
        Full training logic
        """
        self.logger.info(f"\n⏩⏩ Start Training | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ⏩⏩\n"+'=' * 80)
        train_start = time.time()
        for epoch in range(self.start_epoch, self.num_epochs + 1):
             # train one epoch
            epoch_start = time.time()
            self._train_epoch(epoch)
            epoch_end = time.time()

            # save ckp
            if is_master():
                self.save_checkpoint(epoch, save_latest_k=self.save_latest_k, milestone_ckp=self.milestone_ckp)
            
            # log after epoch
            self.logger.info(
                f'🕒 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Epoch Time Cost: {epoch_end-epoch_start:.2f}s, Total Time Cost: {(epoch_end-train_start)/3600:.2f}h\n')
            self.logger.info('=' * 80)
            if self.config.runtime.n_gpus > 1:
                dist.barrier()
        
        # log after training
        self.logger.info(
                f"\n🎉🎉 Finish Training | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}🎉🎉\n\n")

    def save_checkpoint(self, epoch, save_best=False, save_latest_k=-1, milestone_ckp=[]):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, save a copy of current checkpoint file as 'model_best.pth'
        :param save_latest_k: keep only the latest k checkpoints (-1 for keeping all)
        :param milestone_ckp: save and keep current checkpoints if current epoch is in this milestone_ckp
        """
        if self.config.runtime.n_gpus > 1: # ddp
            save_model = self.model.module
        else:
            save_model = self.model
        arch = type(save_model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': save_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }

        filename = opj(self.config.runtime.checkpoints_dir, f'ckp-epoch{epoch}.pth')
        torch.save(state, filename)
        self.logger.info(
            f"💾 Model checkpoint saved at:\n\t{filename}")
        if save_best:
            best_path = opj(self.config.runtime.checkpoints_dir, 'model_best.pth')
            copyfile(filename, best_path)
            self.logger.info(
                f"🔄 Renewing best checkpoint!")
        if milestone_ckp and epoch in milestone_ckp:
            landmark_path = opj(self.config.runtime.checkpoints_dir, f'model_epoch{epoch}.pth')
            copyfile(filename, landmark_path)
            self.logger.info(
                f"🔖 Saving milestone checkpoint at epoch {epoch}!")
        if save_latest_k > 0:
            latest_path = opj(self.config.runtime.checkpoints_dir, 'model_latest.pth')
            copyfile(filename, latest_path)
            outdated_path = opj(self.config.runtime.checkpoints_dir, f'ckp-epoch{epoch-save_latest_k}.pth')
            try:
                os.remove(outdated_path)
            except FileNotFoundError:
                # this happens when current model is loaded from checkpoint
                # or target file is already removed somehow
                pass

    def resume_checkpoint(self, resume_path, resume_conf=['epoch', 'optimizer']):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        :param resume_conf: resume config that controls what to resume
        """

        resume_path = opj(os.getcwd(), self.config.trainer.resume)
        self.logger.info(f"📥 Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path)

        # load architecture params from checkpoint.
        if checkpoint['config'].get('arch', None) != self.config.get('arch', None):
            self.logger.warning("⚠️ Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        
        # preprocess DDP saved cpk
        if checkpoint['config'].get('arch', 1) > 1:
            state_dict = {k.replace('module.', ''): v for k,
                        v in state_dict.items()}
        
        # load cpk
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint
        if 'optimizer' in resume_conf:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info(
                f'📣 Optimizer resumed from the loaded checkpoint!')

        # epoch start point
        if 'epoch' in resume_conf:
            self.start_epoch = checkpoint['epoch'] + 1
            self.logger.info(
                f"📣 Epoch index resumed to epoch ({checkpoint['epoch']}).")
        else:
            self.start_epoch = 1
            self.logger.info(
                f"📣 Epoch index renumbered from epoch (1).")
    
    def writer_update(self, step, phase, metrics, image_tensors=None):
        # writer hook after iter
        self.writer.set_step(step, speed_chk=f'{phase}')

        metric_str = ''
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.writer.add_scalar(f'{phase}/{k}', v)
                metric_str += f'{k}: {v:8.5f} '
        
        if image_tensors:
            for k, v in image_tensors.items():
                self.writer.add_image(
                    f'{phase}/{k}', make_grid(image_tensors[k][0:8, ...].cpu(), nrow=2, normalize=True))
        
        return metric_str # metric string for logger
    
    def progress(self, batch_size, batch_idx, batch_num):
        # base = '[{}/{} ({:.2f}%)]'
        base = '[{:6.2f}%]'
        total = batch_num*batch_size
        current = (batch_idx+1)*batch_size
        if dist.is_initialized():
            current *= dist.get_world_size()
        # proc_str = base.format(current, total, 100.0 * current / total)
        proc_str = base.format(100.0 * current / total)
        return proc_str


    def clip_gradient(self, optimizer, grad_clip=0.5):
        """
        Clips gradients computed during backpropagation to avoid explosion of gradients.

        :param optimizer: optimizer with the gradients to be clipped
        :param grad_clip: clip value
        """
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)