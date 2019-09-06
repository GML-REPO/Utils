import numpy as np
import cv2

def resize_with_pad(img, target_size=512):
    H, W = img.shape[:2]
    C = 0
    
    if len(img.shape) == 3:
        C = img.shape[-1]
    ratio = H/W
    new_H, new_W = H,W
    pad = 0
    
    if ratio > 1:
        new_H = target_size
        new_W = target_size * new_W // H
        pad = target_size - new_W
    elif ratio < 1:
        new_H = target_size * new_H // W
        new_W = target_size
        pad = target_size - new_H
        
    pad = pad // 2

    new_img = cv2.resize(img, (new_W, new_H))

    if C == 0:
        base = np.zeros([target_size, target_size], dtype=img.dtype)
    else:
        base = np.zeros([target_size, target_size, C], dtype=img.dtype)
    
    if ratio > 1:
        base[:, pad:new_W+pad] = new_img
    elif ratio < 1:
        base[pad:pad+new_H, :] = new_img

    return base
    
if __name__ == '__main__':
    img = np.ones([1024,300,3])*255
    new = resize_with_pad(img)
