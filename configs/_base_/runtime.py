# ------------------------------------------------------------------
#  runtime settings
# ------------------------------------------------------------------
runtime = dict(
    exp_name = 'UNet',
    status = 'train',   # 'train' | 'test' | 'realexp'
    gpus = [6],         # gpus to use
    tensorboard=True,   # whether use tensorboard
    work_dir = 'work_dirs/' # work directory
)