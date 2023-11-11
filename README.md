
## [中文](README_cn.md)

# Introduction

This is a code template library for deep learning project development, which has the following features:

* Simple and lightweight configuration system: Based on the native python dict format, with fewer external dependencies, a simple and lightweight project configuration system is implemented, which can easily modify, save and load configurations, thus achieving convenient project management.
* Easily extensible module registration system: Based on the Registry mode similar to the mmcv library, the registration and invocation of various machine learning code modules are implemented, which can easily add, delete and modify modules, thus improving the extensibility of the code library.
* Multi-card training support: Supports pytorch's multi-card training (DDP), which can easily perform multi-card training to improve training speed.

# Installation
Please refer to the [installation instructions](docs/install.md) for installation.

# Detailed Introduction
* [Codebase Introduction](docs/introduction.md)
* [configurations](docs/config.md)
* [Model Training Dataset](docs/add_datasets.md)
* [Add Custom Models](docs/add_models.md)
* [Conversion from Images to Videos and GIFs](docs/video_gif.md)
* [Conversion from PyTorch to ONNX and TensorRT Models](docs/onnx_tensorrt.md)
* [More](docs/)

# Acknowledgments
This codebase is modified from [cacti](https://github.com/ucaswangls/cacti), and thanks to the original author for providing elegant code.