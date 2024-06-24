import argparse
import os 
from tqdm import tqdm
from glob import glob 

from mmseg.apis import init_segmentor, inference_segmentor
from tools.utils import make_cityscapes_format, imread, imwrite, createLayersFromLabel

import cv2
import numpy as np


"""
콘크리트 손상 정보 inference 자료 cityscapes format 으로 생성
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg inference code'
    )
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'img_dir',
        help=('image directory for inference'))
    parser.add_argument(
        'save_dir',
        help=('directory to save result'))
    parser.add_argument(
        '--damage_type',
        default='crack',
        help=('damage type to be inference'))
    

    args = parser.parse_args()

    return args


def get_damage_palette (damage) :
    print(damage)
    if damage == "crack":
        damage = [[0, 0, 0], [255, 0, 0]]
        damage_idx = 1
    if damage == "efflorescence":
        damage = [[0, 0, 0], [0, 255, 0]]
        damage_idx = 2
    if damage == "rebar":
        damage = [[0, 0, 0], [255, 255, 0]]
        damage_idx = 3
    if damage == "spalling":
        damage = [[0, 0, 0], [0, 0, 255]]
        damage_idx = 4
    return damage, damage_idx


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device='cuda:0')

    img_list = glob(os.path.join(args.img_dir, '*.png'))

    damage, damage_idx = get_damage_palette (args.damage_type)

    for img in tqdm(img_list):
        
        gt_path = make_cityscapes_format(
                                         image=img, 
                                         save_dir=args.save_dir, 
                                        )

        src = imread(img)
        src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        cv2.imshow("src", src)
        cv2.waitKey(0)
        
        src_label = imread(gt_path)
        
        result = inference_segmentor(model, src)

        idx = np.argwhere(result[0] == 1)
        y_idx, x_idx = idx[:, 0], idx[:, 1]
        src_label[y_idx, x_idx] = damage_idx
        
        imwrite(gt_path, src_label)
            
        
if __name__ == '__main__':
    main()
