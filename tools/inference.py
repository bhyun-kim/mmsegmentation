import argparse
import os.path as osp 


from glob import glob 

from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette 

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
        'dataset_type',
        help=('data type to be inference'))

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device='cuda:0')

    img_list = glob(osp.join(args.img_dir, '*.png'))

    for img in img_list:
        result = inference_segmentor(model, img)
        out_file = osp.join(args.save_dir,  osp.basename(img))
        print(out_file)
        show_result_pyplot(
            model, img, result, get_palette(args.dataset_type), 
            out_file=out_file)

if __name__ == '__main__':
    main()
