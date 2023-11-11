# Codebase Reference Documentation

## Root Directory
* train.py (entry point for model training)
* test.py (entry point for model testing)
* scripts (some command line scripts)
* checkpoints (path to save pre-trained models)
* datasets (path to datasets)
* docs (detailed documentation)
* requirements.txt (dependencies)

## configs Folder
* __base__ folder: basic configuration for models and data, can be inherited by other configuration files
* Configuration files for various reconstruction algorithms

## srcs Folder
* datasets (specific implementation for data preprocessing)
* models (specific implementation for reconstruction algorithms)
* losses (specific implementation for loss functions)
* metrics (specific implementation for evaluation metrics)
* optimizers (specific implementation for optimizers and learning rate adjustment strategies)
* trainer (specific implementation for trainers)
* tester (specific implementation for testers)
* utils (some common functions, such as PSNR, SSIM calculation)

## toolboxes Folder
* img_proc/ (some common functions for image processing)
* onnx_tensorrt (onnx, tensorrt model conversion and testing)
* video_gif (conversion from images to videos and gifs)
