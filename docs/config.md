
# Project Configuration System

This code template library is based on the native python dict format, with fewer external dependencies, and implements a simple and lightweight project configuration system that can easily modify, save and load configurations, thus achieving convenient project management.

## 1. Organization of Configuration Files

* Configuration files are centrally stored in the `configs` folder, with each sub-configuration folder corresponding to a sub-project/model. The configuration file is a .py file, and the configuration items inside are organized according to the python dictionary type. When loading the configuration, the configuration items in the configuration file will be loaded into a global dictionary, and syntax checks will be automatically performed on issues such as duplicate keys.
* The `configs/_base_/` folder contains basic configuration files, which contain some commonly used configuration items. The configuration files of sub-projects/models can inherit these configuration items. The basic method is to inherit through the `_base_` field. For example, add `_base_ = ['../_base_/default_runtime.py']` to the **header** of the specific .py configuration file, which means to inherit the configuration items in the `configs/_base_/default_runtime.py` file. If the same key appears in the configuration file as the inherited configuration item, it will overwrite the inherited configuration item.

## 2. Passing, Saving and Loading Configuration Files

* When running training or testing code, specify the configuration file through command line commands, such as `python train.py configs/UNet/unet.py --status train`, where command line parameters (such as `--status train` here) will also be passed as configuration items and parsed by argparse in the main function.
* The runtime configuration will be saved in the `config.json` format in the working directory. This configuration file is used to record the configuration of training/testing and can also be used for subsequent code replication (such as `python train.py work_dirs/xxx/config.json`).