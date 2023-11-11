from .pipelines import Compose
from .builder import DATASETS
from .pipelines.builder import build_pipeline
#---------------------------------------
from .utils import get_file_path
from torch.utils.data import Dataset 
import cv2
import numpy as np

# =================
# Datasets
# =================
@DATASETS.register_module 
class NoisyImg(Dataset):
    def __init__(self,data_dir,*args,**kwargs):

        self.data_dir= data_dir
        if "pipeline" in kwargs:
            self.pipeline = Compose(kwargs["pipeline"])
        else:
            self.pipeline = lambda x:x
        self.add_noise = build_pipeline(kwargs["add_noise"])

        # get image paths and load images
        ext = ['jpg', 'png', 'tif', 'bmp']
        self.img_paths, self.img_num, skip_num = get_file_path(data_dir, ext)
        # print(f'---> total dataset image num: {self.img_num}')
        
    def __getitem__(self,index):
        
        img_k = cv2.imread(self.img_paths[index])
        img_k = cv2.cvtColor(img_k, cv2.COLOR_BGR2RGB)
        img_k = img_k.astype(np.float32)/255
        img_k = self.pipeline(img_k)
        img_noisy = self.add_noise(img_k)
        return img_noisy.transpose(2, 0, 1), img_k.transpose(2, 0, 1)
    def __len__(self,):
        return self.img_num

@DATASETS.register_module 
class NoisyImgRealexp(Dataset):
    def __init__(self,data_dir,*args,**kwargs):

        self.data_dir= data_dir

        # get image paths and load images
        ext = ['jpg', 'png', 'tif', 'bmp']
        self.img_paths, self.img_num, skip_num = get_file_path(data_dir, ext)
        # print(f'---> total dataset image num: {self.img_num}')
        
    def __getitem__(self,index):
        
        img_k = cv2.imread(self.img_paths[index])
        img_k = cv2.cvtColor(img_k, cv2.COLOR_BGR2RGB)
        img_k = img_k.astype(np.float32)/255
        img_gt = np.zeros_like(img_k)
        return img_k.transpose(2, 0, 1), img_gt.transpose(2, 0, 1)
    def __len__(self,):
        return self.img_num