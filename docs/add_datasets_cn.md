# 数据集的代码和配置

## 新增数据集

### 1.添加新数据集代码

> 以 `srcs/datasets/float_imgnoise_dataset.py` 为例

新建文件`srcs/datasets/float_imgnoise_dataset.py`文件
```
from .builder import DATASETS

@DATASETS.register_module 
class NoisyImg(Dataset):
    def __init__(self,data_root,...):
        pass
    def __getitem__(self,index):
        pass
```

## 2. 导入该模块
在`srcs/datasets/__init__.py`文件中添加以下代码：
```
from .float_imgnoise_dataset import  NoisyImg
```

## 3. 添加配置文件
新建 `configs/__base__/data.py` 文件，添加相关配置项
```
train_pipeline = [
    dict(type='Flip', direction='horizontal',flip_ratio=0.5,),
    dict(type='Resize',resize_h=256,resize_w=256)，
    ...
] #添加或删除数据增强方法

add_noise = dict(type='AddGaussianNoise', sigma=0.05)

train_dataset = dict(
    type="NoisyImg",
    data_dir="/hhd/2/zzh/project/simple_pytorch_template/datasets/train/Kodak24/",
    pipeline=train_pipeline,
    add_noise = add_noise,
)
```