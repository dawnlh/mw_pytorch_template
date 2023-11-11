from abc import abstractmethod, ABCMeta
from srcs.utils.eval import gpu_inference_time
from ptflops import get_model_complexity_info
from srcs.utils.logger import Logger
import os.path as osp

class BaseTester(metaclass=ABCMeta):
    def __init__(self, model, cfg, metrics, test_dataloader, device):
        self.test_dataloader  = test_dataloader
        self.model = model.to(device)
        self.device = device
        self.metrics = metrics
        self.cfg = cfg
        self.save_img = cfg.tester.get('save_img', False)
        self.logger = Logger(name='tester', log_path=osp.join(cfg.runtime.work_dir,'log.txt'))


        # dataset & model info
        dash_line = '=' * 80 + '\n'
        self.logger.info('Dataset info:\n'
                    + dash_line + 
                    f'num_samples: {len(test_dataloader.dataset)}' +'\n' +
                    dash_line)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        param_num = f'Trainable parameters: {sum([p.numel() for p in trainable_params])/1e6:.2f} M\n'
        self.logger.info('Model info:\n'
                    + dash_line + 
                    str(model)+'\n'+ param_num +'\n'+
                    dash_line)

        # model complexity
        if cfg.tester.model_complexity_meas:
            macs, params = get_model_complexity_info(model=model, verbose=False, print_per_layer_stat=False,**cfg.tester.model_complexity_meas)
            self.logger.info(
                '='*80+'\n{:<30} {}'.format('Inputs resolution: ', cfg.tester.model_complexity_meas.input_res))
            self.logger.info(
                '{:<30} {}'.format('Computational complexity: ', macs))
            self.logger.info('{:<30}  {}\n'.format(
                'Number of parameters: ', params)+'='*80)

        # inference speed
        if cfg.tester.inference_speed_meas:
            gpu_inference_time(model, logger=self.logger, device=device, **cfg.tester.inference_speed_meas)

        
        
    @abstractmethod
    def test(self):
        # realize test function in child class
        raise NotImplementedError

