# 安装说明
## 环境要求
* Python 3
* Pytorch 1.9+
* Numpy
* Opencv-python 
* Scikit-image
* Scikit-learn
* ... (更多依赖请参考 requirements.txt)


## 安装步骤

### 准备安装环境 

Pytorch 安装

> 已经安装了 Pytorch 的可以直接跳过
```
#conda 方式进行安装
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

#pip 方式进行安装
pip install torch torchvision torchaudio
```
* 默认安装最新Pytorch版本，更多Pyotrch版本安装请参考[Pytorch官网](https://pytorch.org)
* 注意Pytorch可用conda或pip任意一种方式进行安装，如果安装速度较慢，可使用pip加清华源的方式进行加速
```
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple 
```
* 在Python环境下执行输入以下代码，验证Pytorch是否安装成功：
```
>>import torch 
>>torch.cuda.is_availble()
#如果输出为True,表示pytorch GPU版本安装成功
```
### 安装模板代码库

```
# 下载代码仓库
git clone https://github.com/dawnlh/simple_pytorch_template

# 安装相关依赖包
cd simple_pytorch_template
pip install -r requirements.txt

# 编译安装（可选）
python setup.py develop
```
* 如果 pip install -r requirements.txt 安装较慢，可添加清华源进行加速

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
```
