import einops
import numpy as np
import cv2


def tensor2uint(img):
    # convert torch tensor to uint
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    return np.uint8((img*255.0).round())

def imsave(img, img_path, compress_ratio=0):
    # function: RGB image saving （H*W*3， numpy）
    # compress_ratio: 1-10, higher value, lower quality
    # tip: default $compress_ratio for built-in function cv2.imwrite is 95/100 (higher value, higher quality) and 3/10 (higher value, lower quality) for jpg and png foramt, respectively. Here default value set to no compression
    
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
        img = img[:, :, ::-1]
    if img_path.split('.')[-1] in ['jpg', 'jpeg']:
        cv2.imwrite(img_path, img, [
                    cv2.IMWRITE_JPEG_QUALITY, (10-compress_ratio)*10])
    elif img_path.split('.')[-1] in ['png', 'PNG']:
        cv2.imwrite(img_path, img, [
                    cv2.IMWRITE_PNG_COMPRESSION, compress_ratio])
    else:
        cv2.imwrite(img_path, img)

def imsave_n(imgs:list,img_path,axis=1,show_flag=False):
    # save one or more images (np.ndarray, [c,h,w]) to one image
    if imgs[0].ndim==3:
        # rgb images
        imgs = [np.transpose(img, (1, 2, 0)) for img in imgs]
        result_img = np.concatenate(imgs,axis=axis)
        result_img = result_img[:,:,::-1]
    elif imgs[0].ndim==2:
        # gray images
        result_img = np.concatenate(imgs,axis=axis)
    
    cv2.imwrite(img_path,result_img)
    
    if show_flag:
        cv2.namedWindow("image",0)
        cv2.imshow("image",result_img.astype(np.uint8))
        cv2.waitKey(0)

def vidsave_n(imgs:list,img_path,axis=0,show_flag=False):
    # save one or more videos (np.ndarray [n,c,h,w]) to one image
    if imgs[0].ndim==4:
        # rgb videos
        for i in range(len(imgs)):
            imgs[i] = einops.rearrange(imgs[i],"n c h w->h (n w) c")
        result_img = np.concatenate(imgs,axis=axis)
        result_img = result_img[:,:,::-1]
    elif imgs[0].ndim==3:
        # gray videos
        for i in range(len(imgs)):
            imgs[i] = einops.rearrange(imgs[i],"n h w->h (n w)")
        result_img = np.concatenate(imgs,axis=axis)
    
    cv2.imwrite(img_path,result_img)
    
    if show_flag:
        cv2.namedWindow("image",0)
        cv2.imshow("image",result_img.astype(np.uint8))
        cv2.waitKey(0)