# ------------------------------------------------------------------
# import from _base_
# ------------------------------------------------------------------

# _base_=[
#         "../_base_/runtime.py",
#         "../_base_/network.py",
#         "../_base_/data.py",
#         "../_base_/trainer.py",
#         "../_base_/tester.py"
#         ]

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

# ------------------------------------------------------------------
# network settings
# ------------------------------------------------------------------

# architecture
model = dict(
    type='UNet',
    in_nc=3,
    out_nc=3
)
# loss = dict(type='MSELoss')
loss = dict(type='WeightedLoss', loss_conf_dict={'L1Loss':0.5, 'MSELoss':0.3, 'TVLoss':0.2})

# optimizer & lr_scheduler
optimizer = dict(type='Adam', lr=0.0001)
lr_scheduler = dict(type='StepLR', step_size=10, gamma=0.5)

# metrics
metric = dict(type='IQA_Metric', metric_names=['psnr', 'ssim'], calc_mean=True)


# ------------------------------------------------------------------
# dataset settings
# ------------------------------------------------------------------

# pipeline for image preprocessing
resize_h,resize_w = 256,256
train_pipeline = [ 
    dict(type='RandomResize'),
    dict(type='RandomCrop',crop_h=resize_h,crop_w=resize_w,random_size=True),
    dict(type='Flip', direction='horizontal',flip_ratio=0.5,),
    dict(type='Flip', direction='diagonal',flip_ratio=0.5,),
    dict(type='Resize', resize_h=resize_h,resize_w=resize_w),
]
add_noise = dict(type='AddGaussianNoise', sigma=0.05)

# dataset
train_dataset = dict(
    type="NoisyImg",
    data_dir="/hhd/2/zzh/project/simple_pytorch_template/datasets/train/Kodak24/",
    pipeline=train_pipeline,
    add_noise = add_noise,
)

test_dataset = dict(
    type="NoisyImg",
    data_dir="/hhd/2/zzh/project/simple_pytorch_template/datasets/train/Kodak24/",
    add_noise = add_noise,
)

# dataloader
train_dataloader = dict(
    dataset = train_dataset,
    val_part=0.1, # float | val_dataset
    batch_size=4,
    num_workers = 4
    )
test_dataloader = dict(
    dataset = test_dataset, # test_dataset | simuexp_dataset | realexp_dataset
    batch_size=4,
    num_workers = 4
    )

# ------------------------------------------------------------------
# trainer settings
# ------------------------------------------------------------------

trainer = dict(
    name = 'trainer', # name of the trainer file in srcs/trainer
    num_epochs=20,    # total epochs to run
    resume = None,    # path of the checkpoint to resume
    resume_conf = ['epoch', 'optimizer'], # resume config
    save_latest_k=5,   # save the latest k checkpoint
    milestone_ckp = [], # save the checkpoint at the milestone epoch
    logging_interval=2, # log interval
    eval_interval=1, # eval interval
    max_iter=None # limit max_iter for each epoch
    )

# ------------------------------------------------------------------
# tester settings
# ------------------------------------------------------------------

inference_speed_meas = dict(
    input_shape=(1, 3, 256, 256),
    repetitions=100
    )

model_complexity_meas = dict(
    input_res=(3, 256, 256),
    input_constructor=None
    )

tester = dict(
    name = 'tester', # name of the tester file in srcs/tester
    save_img = True, # whether save the input and output image
    model_complexity_meas = model_complexity_meas, 
    inference_speed_meas = inference_speed_meas,
    checkpoint = '/hhd/2/zzh/project/simple_pytorch_template/work_dirs/UNet/train/2023-11-11_23-24-26/checkpoints/model_latest.pth'
)
