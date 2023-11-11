# Add Custom Models
## 1. Add code for custom models (using Unet network as an example):

Create `srcs/models/unet.py` 
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

## 2. Import the module
Add the following codes in `srcs/models/__init__.py`：
```
from .unet import UNet
```
## 3. Add configuration file
Create `configs/__base__/network.py` file and add the following configuration items
```
model = dict(
    type='UNet',
    in_nc=3,
    out_nc=3
)
```

## 4. Configuration for optimizers and lr_schedulers
Similar to adding new custom models, not repeated here.