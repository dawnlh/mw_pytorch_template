train_pipeline = [
    dict(type='Flip', direction='horizontal',flip_ratio=0.5,),
    dict(type='Resize',resize_h=256,resize_w=256)
]

gene_meas = dict(type='GenerationGrayMeas')

train_data = dict(
    type="DavisData",
    # data_root="/media/wangls/new_disk/datasetes/SCI/DAVIS/DAVIS-480/JPEGImages/480p",
    data_root="E:/datasetes/SCI/DAVIS/DAVIS-480/JPEGImages/480p",
    mask_path="test_datasets/mask/mask.mat",
    pipeline=train_pipeline,
    gene_meas = gene_meas,
    mask_shape = None
)
