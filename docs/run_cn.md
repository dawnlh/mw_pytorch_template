# 代码运行

## 训练

支持高效多GPU训练与单GPU训练, 首先根据把模型、数据集的代码实现和配置文件写好。

多GPU训练可通过以下方式进行启动：

### 1. 通过配置文件设置 （推荐）

在配置文件中指定 `config.runtime.gpus=[0,1,2,3]`，表示使用0号到3号显卡进行训练，然后执行以下命令启动训练：

```
python train.py configs/UNet/unet.py
```

### 2. 通过命令行设置

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 scripts/train.py configs/UNet/unet.py --distributed=True
```
* 其中 CUDA_VISIBLE_DEVICES 指定显卡编号  
* --nproc_per_node 表示使用显卡数量  
* --master_port 表示主节点端口号,主要用于通信

注意：这种方式是把所有训练/测试过程代码都在一个文件中实现（不依赖于`srcs/trainer`和`srcs/tester`），因此只适合简单固定的训练/测试过程，比如单任务/单模型的训练/测试。如果需要更多元、模块化的训练/测试过程及更多功能，建议使用配置文件的方式。

单GPU训练可通过以下方式进行启动，默认为0号显卡，也可通过设置CUDA_VISIBLE_DEVICES编号选择显卡：

```
python scripts/train.py configs/UNet/unet.py
```

## 测试

在配置文件中设置好测试的一些参数，比如数据集、预训练模型、测试项目等。然后执行以下命令启动测试：

```
python scripts/test.py configs/UNet/unet.py --checkpoint checkpoints/unet/unet.pth
```

* 注意：权重参数路径可以通过 --checkpoint 进行指定，也可以修改配置文件中`config.tester.checkpoint`值
* 测试状态`config.runtime.status`有四种类型，即 'train', 'test', 'realexp', 'simuexp'， 分别表示普通的训练、测试、以及实采实验、仿真实验。后面二者主要在做benchmark实验时使用，可以作为传递给 dataset代码的参数，来控制数据集的生成和加载方式。

