
## tester
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
