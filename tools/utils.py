import os

import cv2
import numpy as np

def imread(imgPath):
    img = cv2.imdecode(np.fromfile(imgPath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img.ndim == 3 : 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def imwrite(path, img): 
    _, ext = os.path.splitext(path)
    _, label_to_file = cv2.imencode(ext, img)
    label_to_file.tofile(path)


def createLayersFromLabel(label, num_class):

    layers = []

    for idx in range(num_class):
        layers.append(label == idx)
        
    return layers



def make_cityscapes_format (image, save_dir) :
    temp_img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    gt = np.zeros((temp_img.shape[0], temp_img.shape[1]), dtype=np.uint8)

    
    
    save_dir_img = os.path.join(save_dir, 'leftImg8bit')
    save_dir_gt = os.path.join(save_dir, 'gtFine')

    os.makedirs(save_dir_img, exist_ok=True)
    os.makedirs(save_dir_gt, exist_ok=True)

    img_filename = os.path.basename(image)
    img_filename = img_filename.replace('.png', '_leftImg8bit.png')
    
    gt_filename = img_filename.replace('_leftImg8bit.png', '_gtFine_labelIds.png')

    is_success, org_img = cv2.imencode(".png", temp_img)
    org_img.tofile(os.path.join(save_dir_img, img_filename))

    is_success, gt_img = cv2.imencode(".png", gt)
    gt_img.tofile(os.path.join(save_dir_gt, gt_filename))
    gt_path = os.path.join(save_dir_gt, gt_filename) 
    
    return gt_path

def inference_cityscapes (org_img, gt_img):
    print("d")
