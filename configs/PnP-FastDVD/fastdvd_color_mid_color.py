_base_=["../_base_/matlab_bayer.py"]
test_data = dict(
    data_root="test_datasets/middle_scale",
    mask_path="test_datasets/mask/mid_color_mask.mat",
    rot_flip_flag=True
)

model = dict(
    type='FastDVDnet',
    num_input_frames=5, 
    num_color_channels=3
)

denoise_method="GAP"
checkpoints="checkpoints/fastdvd/fastdvd_color.pth"

sigma_list = [50/255, 25/255, 12/255]
iter_list = [50, 30, 10] 
show_flag=True
demosaic = True
color_denoiser=True
use_cv2_demosaic=False
