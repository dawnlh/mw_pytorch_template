from ._base_trainer import BaseTrainer
import torch
from ._utils import collect,is_master
from torchvision.utils import make_grid
from ._base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, optimizer, criterion, metrics, lr_scheduler, config,  train_dataloader, val_dataloader=None):
        super(Trainer, self).__init__(model, optimizer, criterion, metrics, lr_scheduler, config,  train_dataloader, val_dataloader)
        # optimizer gradient clip value
        self.grad_clip = 0.5  


    def writer_update(self, step, phase, metrics, image_tensors=None):
        # hook after iter
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

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        """

        ## train
        self.model.train()
        for batch_idx, (img_noise, img_target) in enumerate(self.train_dataloader):
            img_noise, img_target = img_noise.to(
                self.device), img_target.to(self.device)

            # forward
            output = self.model(img_noise)

            # loss calc
            loss = self.criterion(output, img_target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            # clip gradient and update
            self.clip_gradient(self.optimizer, self.grad_clip)
            self.optimizer.step()

            # iter record
            if batch_idx % self.logging_interval == 0 or (batch_idx+1) == self.max_iter:
                # iter metrics
                iter_metrics = {}
                # average metric between processes
                calc_metrics = self.metrics(output, img_target)
                # add loss to iter_metrics
                calc_metrics.update({'loss':loss})
                for k, v in calc_metrics.items():
                    vv = collect(v) if self.config.runtime.n_gpus > 1 else v
                    iter_metrics.update({k: vv})

                # logger & writer
                metric_str = self.writer_update(epoch*self.max_iter+batch_idx, '[train]',iter_metrics, {}) 
                proc_str = self.progress(self.config.train_dataloader.batch_size, batch_idx, self.max_iter)
                if is_master():
                    self.logger.info(f'Train Epoch: {epoch:03d} {proc_str}  lr: {self.optimizer.param_groups[0]["lr"]:.4e} | {metric_str}')
            
            # max iter stop
            if (batch_idx+1) == self.max_iter:
                break
        
        ## valid
        if self.val_dataloader and (epoch % self.eval_interval == 0):
            self._valid_epoch(epoch)

        ## learning rate update
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        iter_metrics = {}
        with torch.no_grad():
            # valid iter
            for batch_idx, (img_noise, img_target) in enumerate(self.val_dataloader):
                img_noise, img_target = img_noise.to(
                    self.device), img_target.to(self.device)
                
                # forward
                output = self.model(img_noise)
                # loss calc
                loss = self.criterion(output, img_target)
                
                # metrics
                calc_metrics = self.metrics(output, img_target)
                calc_metrics.update({'loss':loss}) # add loss to iter_metrics
                
                # average metric between processes
                for k, v in calc_metrics.items():
                    vv = collect(v) if self.config.runtime.n_gpus > 1 else v
                    iter_metrics.update({k:iter_metrics.get(k, 0)+vv})
                
            # average iter metrics
            iter_metrics = {k:v/(batch_idx+1) for k,v in iter_metrics.items()}
            # iter images
            image_tensors = {
                'input': img_noise[0:4, ...], 'img_target': img_target[0:4, ...], 'output': output[0:4, ...]}

            # logger & writer
            metric_str = self.writer_update(epoch, '[valid]', iter_metrics, image_tensors)
            if is_master():
                self.logger.info(
                '-'*80 + f'\nValid Epoch: {epoch} [{epoch:03d}/{self.num_epochs:03d}]  lr: {self.optimizer.param_groups[0]["lr"]:.4e} | {metric_str}\n' + '-'*80)