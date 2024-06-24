import argparse
import os.path as osp 


from glob import glob 

from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot, save_result_palette

from tqdm import tqdm

"""
콘크리트 손상 정보 inference
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
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--show',
        default=True,
        help='Show the result images')
    

    args = parser.parse_args()

    return args


def get_damage_palette (damage) :
    print(damage)
    if damage == "crack":
        damage = [[0, 0, 0], [255, 0, 0]]
    if damage == "efflorescence":
        damage = [[0, 0, 0], [0, 255, 0]]
    if damage == "rebar":
        damage = [[0, 0, 0], [255, 255, 0]]
    if damage == "spalling":
        damage = [[0, 0, 0], [0, 0, 255]]
    return damage


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device='cuda:0')

    img_list = glob(osp.join(args.img_dir, '*.png'))

    for img in tqdm(img_list):
        result = inference_segmentor(model, img)
        out_file = osp.join(args.save_dir,  osp.basename(img))
        
        if args.show == True :
            show_result_pyplot(
                model, 
                img, 
                result, 
                palette = get_damage_palette(args.damage_type), 
                opacity=args.opacity, 
                out_file=out_file
                )

        elif args.show == None:
            save_result_palette(
                model, 
                img, 
                result,
                palette = get_damage_palette(args.damage_type), 
                opacity=args.opacity, 
                out_file=out_file
            )
if __name__ == '__main__':
    main()
