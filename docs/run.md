# Running

## Training
Support multi GPUs and single GPU training efficiently, first implement model, dataset and configuration file.

Launch multi GPU training by the statement below:

### 1. Set by configuration file

Specify `config.runtime.gpus=[0,1,2,3]` in configuration file, which means using GPU 0 to 3 for training, then launch training by executing the statement below.

```
python train.py configs/UNet/unet.py
```

### 2. Set by command line

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 scripts/train.py configs/UNet/unet.py --distributed=True
```
* CUDA_VISIBLE_DEVICE: specify number of GPUs
* --nproc_per_node: number of used GPUs
* --master_port: main node port number, usually for communication

Notice: this way is to implement all training/testing process in one file (not rely on `srcs/trainer` and `srcs/tester`), so it is only suitable for simple and fixed training/testing process, such as single task/single model training/testing. If you need more diverse and modular training/testing process and more functions, it is recommended to use configuration file.

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python scripts/train.py configs/UNet/unet.py
```

## Testing

Set testing parameters in configuration file, such as dataset, pre-trained model, testing project, etc. Launch testing by executing the statement below.

```
python scripts/test.py configs/UNet/unet.py --checkpoint checkpoints/unet/unet.pth
```

* Note: weight path can be specified by --checkpoint, or modify `config.tester.checkpoint` in configuration file.
* Testing status `config.runtime.status` has four types, i.e. 'train', 'test', 'realexp', 'simuexp', which represent normal training, testing, real experiment and simulation experiment respectively. The last two are mainly used in benchmark experiments, which can be used as parameters passed to dataset code to control the generation and loading of datasets.