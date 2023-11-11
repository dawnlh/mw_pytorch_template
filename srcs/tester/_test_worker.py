from srcs.models.builder import build_model
from srcs.datasets.builder import build_dataloader 
from srcs.utils.logger import Logger
import os.path as osp
import torch

## test worker
def test_worker(Tester, cfg):
    # get logger
    logger = Logger(name='tester', log_path=osp.join(cfg.runtime.work_dir,'log.txt'))

    # build dataloader
    test_dataloader = build_dataloader(cfg)

    
    # build metrics
    # `import` should be here, or os.environ["CUDA_VISIBLE_DEVICES"] will not work
    from srcs.metrics.builder import build_metric
    metrics = build_metric(cfg.metric)

    # load checkpoint
    logger.info(f"📥 Loading checkpoint:\n\t{cfg.tester.checkpoint} ...")
    checkpoint = torch.load(cfg.tester.checkpoint)
    logger.info(f"💡 Checkpoint loaded: epoch {checkpoint['epoch']}.")

    # select cfg file
    if 'cfg' in checkpoint:
        loaded_cfg = checkpoint['cfg']
    else:
        loaded_cfg = cfg

    # build model
    model = build_model(cfg.model)
    
    # load weight
    state_dict = checkpoint['state_dict']
    if len(loaded_cfg.runtime.gpus)>1:
        # preprocess DDP saved cpk
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # DP
    if cfg.runtime.distributed:
        model = torch.nn.DataParallel(
            model, device_ids=list(range(cfg.n_gpus)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # test
    tester = Tester(model, cfg, metrics, test_dataloader, device)
    tester.test()


