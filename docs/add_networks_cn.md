# 新增自定义模型

## 1. 添加自定义模型的代码（以Unet网络为例）： 

新建 `srcs/models/unet.py` 文件
```
from torch import nn 
from .builder import MODELS

@MODELS.register_module
class UNet(nn.Module):
    def __init__(self,args):
        pass
    def forward(self,x):
        pass
```

## 2. 导入该模块
在 `srcs/models/__init__.py` 文件中添加以下代码：
```
from .unet import UNet
```
## 3. 添加配置文件
新建 `configs/__base__/network.py` 文件，添加相关配置项
```
model = dict(
    type='UNet',
    in_nc=3,
    out_nc=3
)
```

## 4.optimizer和lr_scheduler的配置
和添加自定义的新模型类似，不再赘述。