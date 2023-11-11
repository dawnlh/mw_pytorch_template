import os
import torch
import time
from ._base_tester import BaseTester
from tqdm import tqdm
from srcs.utils.image import tensor2uint, imsave_n, imsave


class Tester(BaseTester):
    """
    Tester class
    """
    def __init__(self, model, cfg, metrics, test_dataloader, device):
        super(Tester, self).__init__(model, cfg, metrics, test_dataloader, device)

    
    def test(self):
        # init
        self.model.eval()
        if self.save_img:
            os.makedirs(self.cfg.runtime.output_dir+'/images')

        # run test
        total_metrics = {}
        time_start = time.time()
        with torch.no_grad():
            for i, (img_noise, img_target) in enumerate(tqdm(self.test_dataloader, desc='⏳ Testing')):
                img_noise, img_target = img_noise.to(self.device), img_target.to(self.device)

                # inference
                output = self.model(img_noise)

                # save image
                if self.save_img:
                    for k, (in_img, out_img, gt_img) in enumerate(zip(img_noise, output, img_target)):
                        in_img = tensor2uint(in_img)
                        out_img = tensor2uint(out_img)
                        gt_img = tensor2uint(gt_img)
                        imgs = [in_img, out_img, gt_img]
                        imsave_n(imgs, f'{self.cfg.runtime.output_dir}/images/test{i+1:03d}_{k+1:03d}.png')
                        # imsave(in_img, f'{self.cfg.runtime.output_dir}/images/test{i+1:03d}_{k+1:03d}_in.png')

                # computing metrics on test set (if gt is available)
                if self.cfg.runtime.status != 'realexp':
                    batch_size = img_noise.shape[0]
                    calc_metrics = self.metrics(output, img_target)
                    for k, v in calc_metrics.items():
                        total_metrics.update({k:total_metrics.get(k,0) + v * batch_size})
        # time cost
        time_end = time.time()
        time_cost = time_end-time_start
        n_samples = len(self.test_dataloader.sampler)
        
        # metrics average
        test_metrics = {k: v / n_samples for k, v in total_metrics.items()}
        metrics_str = ' '.join([f'{k}: {v:6.4f}' for k, v in test_metrics.items()])

        self.logger.info('='*80 + f'\n🎯 time/sample {time_cost/n_samples:6.4f} ' + metrics_str + '\n' + '='*80)