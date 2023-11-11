## [English](README.md) 

# 简介

 这是一个用于深度学习项目开发的代码模板库，主要有以下特点：

* 简洁轻便的配置系统：基于原生的 python dict 格式，以较少的外部依赖，实现了简洁轻便的项目配置系统，可以方便的进行配置的修改、保存和加载，从而实现方便的项目管理。
* 易扩展的模块注册系统：基于类似于 mmcv 库的 Registry 模式，实现了机器学习代码各个模块的注册与调用，可以方便的进行模块的添加、删除和修改，从而提升代码库的可扩展性。
* 多卡训练支持：支持 pytorch 的多卡训练（DDP），可以方便的进行多卡训练，从而提升训练速度。

# 安装
请参考[安装说明文档](docs/install_cn.md)进行安装

# 详细介绍
* [CACTI代码库说明文档](docs/introduction_cn.md)
* [配置系统](docs/config_cn.md)
* [模型训练数据集](docs/add_datasets_cn.md)
* [新增自定义模型](docs/add_models_cn.md)
* [图片到视频与gif的转换](docs/video_gif_cn.md)
* [pytorch到onnx与tensorrt模型的转换](docs/onnx_tensorrt_cn.md)
* [更多](docs/)


# 致谢
本代码库是从[cacti](https://github.com/ucaswangls/cacti)修改而来，感谢原作者提供的优雅代码。
