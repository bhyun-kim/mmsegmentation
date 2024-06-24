# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class CityscapesDataset_Efflorescence(CityscapesDataset):
    """Cityscapes dataset with Single Class.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    CLASSES = ('background', 'efflorescence')

    PALETTE = [[0, 0, 0], [0, 255, 0]]

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelIds.png',
                 **kwargs):
        super(CityscapesDataset_Efflorescence, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

