# Dataset code and configuration

## Add new dataset

### 1. Add new dataset code

>  `srcs/datasets/float_imgnoise_dataset.py` is used as an example

Create `srcs/datasets/float_imgnoise_dataset.py`
```
from .builder import DATASETS

@DATASETS.register_module 
class NoisyImg(Dataset):
    def __init__(self,data_root,...):
        pass
    def __getitem__(self,index):
        pass
```

## 2. Import the module
Add the following codes in  `srcs/datasets/__init__.py`：
```
from .float_imgnoise_dataset import  NoisyImg
```

## 3. Add configuration file
Create `configs/__base__/data.py` and add the following codes：
```
train_pipeline = [
    dict(type='Flip', direction='horizontal',flip_ratio=0.5,),
    dict(type='Resize',resize_h=256,resize_w=256)，
    ...
] # add or delete data augmentation methods

add_noise = dict(type='AddGaussianNoise', sigma=0.05)

train_dataset = dict(
    type="NoisyImg",
    data_dir="/hhd/2/zzh/project/simple_pytorch_template/datasets/train/Kodak24/",
    pipeline=train_pipeline,
    add_noise = add_noise,
)
```