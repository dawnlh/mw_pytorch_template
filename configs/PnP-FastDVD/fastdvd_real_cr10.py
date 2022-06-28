_base_=[
    "../_base_/real_data.py",
]
real_data = dict(
    data_root="test_datasets/real_data/cr10",
    cr=10
)

model = dict(
    type='FastDVDnet',
    num_input_frames=5, 
    num_color_channels=1
)

denoise_method="GAP"
checkpoints="checkpoints/fastdvd/fastdvd_gray.pth"

sigma_list = [100/255,75/255,50/255, 25/255]
iter_list = [30, 20, 20,10] 
show_flag=True
