import random
import cv2
import numpy as np

from .builder import PIPELINES

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC
}
@PIPELINES.register_module
class Resize:
    def __init__(self, resize_h, resize_w, interpolation='bilinear'):
        
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.interpolation = interpolation

    def __call__(self, img):
        resize_img = cv2.resize(img,dsize=(self.resize_w,self.resize_h),
                        interpolation=cv2_interp_codes[self.interpolation])
        return resize_img

@PIPELINES.register_module
class RandomRotation:
    def __init__(self, degrees, rotation_ratio=0.5):
        if degrees<0:
            degrees=-degrees
        self.degrees = (-degrees, degrees)
        self.rotation_ratio=rotation_ratio

    def __call__(self, img):
        rotation = np.random.random() < self.rotation_ratio
        if not rotation:
            return img
        angle = random.uniform(self.degrees[0], self.degrees[1])
        img_dim = img.shape
        if len(img_dim)==3:
            h,w,c = img_dim
        else:
            h,w = img_dim

        center = ((w - 1) * 0.5, (h - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(center, -angle)
        rotated_img = cv2.warpAffine(img, matrix, (w, h))
        return rotated_img

@PIPELINES.register_module
class Flip:
    def __init__(self, direction='horizontal',flip_ratio=0.5):
        _directions = ['horizontal', 'vertical','diagonal']
        assert direction in _directions,"flip direction not define!" 
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __call__(self, img):
        flip = np.random.random() < self.flip_ratio
        if not flip:
            return img
        if self.direction == 'horizontal':
            flag = 1
        elif self.direction == 'vertical':
            flag = 0
        else:
            flag = -1
        flip_img = cv2.flip(img, flag)
        return flip_img

@PIPELINES.register_module
class RandomResize:
    def __init__(self,scale=(0.8,1.2),interpolation='bilinear',resize_ratio=0.5):
        self.scale = scale
        self.interpolation = interpolation
        self.resize_ratio = resize_ratio
    def __call__(self, img):
        resize = np.random.random() < self.resize_ratio
        if not resize:
            return img
        resize_scale = random.uniform(self.scale[0], self.scale[1])
        resize_img = cv2.resize(img, dsize=None,fx=resize_scale,fy=resize_scale,\
                interpolation=cv2_interp_codes[self.interpolation])
        return resize_img

@PIPELINES.register_module
class RandomCrop:
    def __init__(self,crop_h,crop_w,random_size=False,crop_ratio=0.5):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.random_size=random_size
        self.crop_ratio = crop_ratio
    def __call__(self, img):
        crop = np.random.random() < self.crop_ratio
        if not crop:
            return img
        img_dim = img.shape
        if len(img_dim)==3:
            img_h,img_w,_ = img_dim
        else:
            img_h,img_w = img_dim
        assert self.crop_h<img_h or self.crop_w<img_w, \
            "Crop height or width greater than image size! "
        crop_h, crop_w = self.crop_h, self.crop_w
        if self.random_size:
            crop_h = np.random.randint(self.crop_h//2,img_h)
            crop_w = np.random.randint(self.crop_w//2,img_w)
        h_b = np.random.randint(0,img_h-crop_h+1)
        w_b = np.random.randint(0,img_w-crop_w+1)
        crop_img = img[h_b:h_b+crop_h,w_b:w_b+crop_w]
        return crop_img
     