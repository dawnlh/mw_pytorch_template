import cv2 
import numpy as np 
import einops
from .builder import PIPELINES

@PIPELINES.register_module
class AddGaussianNoise:
    """
    Add gaussian noise to image 
    """
    def __init__(self,sigma=0.01):
        self.sigma=sigma

    def __call__(self, img):
        # img: [H,W,C], np.float32
        if isinstance(img,np.int8):
            img = img.astype(np.float32)/255
        img_noise = img + np.random.normal(0, self.sigma, img.shape)
        return img_noise.astype(np.float32)
