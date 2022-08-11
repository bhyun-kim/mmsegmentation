# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler as _DistributedSampler

import numpy as np

from copy import deepcopy

from mmseg.core.utils import sync_random_seed
from mmseg.utils import get_device


class ClassUniformSampler(_DistributedSampler):
    """ClassUniformSampler inheriting from
    `torch.utils.data.DistributedSampler`.

    Args:
        datasets (Dataset): the dataset will be loaded.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, world_size is retrieved from the
            current distributed group.
        rank (int, optional):  Rank of the current process within num_replicas.
            By default, rank is retrieved from the current distributed group.
        shuffle (bool): If True (default), sampler will shuffle the indices.
        seed (int): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
    """

    def __init__(self,
                 dataset: Dataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed=0) -> None:
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        device = get_device()
        self.seed = sync_random_seed(seed, device)

    def __iter__(self) -> Iterator:
        """
         Yields:
            Iterator: iterator of indices for rank.
        """
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            # When :attr:`shuffle=True`, this ensures all replicas
            # use a different random ordering for each epoch.
            # Otherwise, the next iteration of this sampler will
            # yield the same ordering.
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        self.dataset.get_gt_statistics()
        
        self.img_list_per_class = list()
        for i in range(len(self.dataset.CLASSES)):
            self.img_list_per_class.append(list())

        self.num_obj_in_class = dict.fromkeys(self.dataset.CLASSES, 0)

        for idx in indices : 
            img_info = self.dataset.img_infos[idx]
            for class_idx in img_info['ann']['class_info']:
                if class_idx != 255:
                    self.img_list_per_class[class_idx].append(idx)
                    self.num_obj_in_class[self.dataset.CLASSES[class_idx]] += 1


        self.num_obj_in_class = sorted(self.num_obj_in_class.items(), key=lambda x: x[1], reverse=True)

        _indices = torch.arange(len(self.dataset)).tolist()

        indices_return = []
        
        temp_class_list = deepcopy(self.img_list_per_class)

        while np.unique(indices_return).tolist() != _indices:
            
            classes_not_loaded = list(self.dataset.CLASSES)

            while len(classes_not_loaded) > 0 :
                # get the most frequent classes 
                for_most_freq_class = []
                for num_obj in self.num_obj_in_class: 
                    if num_obj[0] in classes_not_loaded:
                        for_most_freq_class.append(num_obj)
                    
                most_freq_class = for_most_freq_class[0][0]
                # print(f"most_freq_class {most_freq_class}")
                class_idx = self.dataset.CLASSES.index(most_freq_class)
                # print(f"class_idx {class_idx}")

                # print(f"temp_class_list {temp_class_list}")

                # 
                idx_to_be_loaded = temp_class_list[class_idx][0]
                indices_return.append(idx_to_be_loaded)

                classes_in_idx_to_be_loaded = [] 

                for class_info in self.dataset.img_infos[idx_to_be_loaded]['ann']['class_info']: 
                    if class_info != 255:
                        classes_in_idx_to_be_loaded.append(self.dataset.CLASSES[class_info])

                # print(f"classes_not_loaded {classes_not_loaded}")

                classes_not_loaded = [x for x in classes_not_loaded if (x not in classes_in_idx_to_be_loaded)]

                # print(f"classes_not_loaded {classes_not_loaded}")

                for idx, class_list in enumerate(temp_class_list):
                    if idx_to_be_loaded in class_list: 
                        temp_class_list[idx].remove(idx_to_be_loaded)
                    if len(temp_class_list[idx]) == 0 : 
                        temp_class_list[idx] += (self.img_list_per_class[idx])

        indices = indices_return

        # add extra samples to make it evenly divisible
        if len(indices) > self.total_size : 
            self.num_samples = len(indices) // self.num_replicas + 1
            self.total_size = self.num_samples*self.num_replicas
            
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # print(f"indices {indices}")
        # print(f"len(indices) {len(indices)}")

        return iter(indices)
