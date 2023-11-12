# ----------------------------------------
# dataset settings
# ----------------------------------------

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

val_dataset = dict(
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

simuexp_dataset = dict(
    type="NoisyImg",
    data_dir="/hhd/2/zzh/project/simple_pytorch_template/datasets/train/Kodak24/",
    add_noise = add_noise,
)

realexp_dataset = dict(
    type="NoisyImgRealexp",
    data_dir="/hhd/2/zzh/project/simple_pytorch_template/datasets/train/Kodak24/",
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
