# Installation Documentation 
## Environment
* Python 3
* Pytorch 1.9+
* Numpy
* Opencv-python 
* Scikit-image
* Scikit-learn
* ... (See requirements.txt for more details)

## Installation Precdure
### Package Installation

Pytorch Installation

> skip if Pytorch already exsit

```
#conda
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

#pip
pip install torch torchvision torchaudio
```

* Install latest Pytorch as default, for different verision please see pytorch.org
* Notice that Pytorch can be installed via conda or pip, if you suffering from network issues, please use pip with Tsinghua index url.

```
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple 
```

* To validate Pytorch installation, type the statements below.
```
>>import torch 
>>torch.cuda.is_availble()
#Get "True" when pytorch (GPU Version) successfully installed 
```

### Code template Installation

```
# Download Code Library
git clone https://github.com/dawnlh/simple_pytorch_template

#Install releative packages
cd simple_pytorch_template
pip install -r requirements.txt

#Compile and install (optional)
python setup.py develop
```
* If you suffering from network issues when running pip install -r requirements.txt command, please use the Tsinghua index url.

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
```
