# 代码库参考说明文档

## 根目录
* train.py （模型训练入口）
* test.py （模型测试入口）
* scripts （一些命令行脚本）
* checkpoints （预训练模型保存路径）
* datasets （数据集路径）
* docs （详细说明文档）
* requirements.txt （依赖包）

## configs文件夹
* __base__文件夹：模型和数据的一些基本配置，可以被其他配置文件继承
* 各个重建算法的配置文件

## srcs文件夹
* datasets （数据预处理的具体实现）
* models （重建算法的具体的实现）
* losses （损失函数的具体实现）
* metrics （评价指标的具体实现）
* optimizers （优化器及学习率调整策略的具体实现）
* trainer （训练器的具体实现）
* tester （测试器的具体实现）
* utils （一些通用函数，如PSNR,SSIM值的计算）

## toolboxes文件夹
* img_proc/ （图像处理的一些通用函数）
* onnx_tensorrt （onnx, tensorrt 模型转换与测试）
* video_gif （图像到视频与动态图的转换）

