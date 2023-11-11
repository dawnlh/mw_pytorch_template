## network
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
