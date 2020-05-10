import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path

import cv2
from .video_process import *
from .videotransforms import *


from torchvision import datasets, transforms


class Danmus(data_utl.Dataset):

    def __init__(self, file, num, record_file=None, mode='rgb', transforms=None, dataset='data1'):
        if dataset == 'data1':
            self.data = make_dataset(file,num)
        elif dataset == 'data2':
            self.data = make_dataset_sta(file,record_file,num)
        else:
            self.data = make_dataset_tos(file, num)
        self.transforms = transforms
        self.file = file
        self.mode = mode
        self.dataset = dataset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        clip, vid = self.data[index]

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.file, vid,clip, dataset=self.dataset)
        else:
            imgs = load_flow_frames(self.file, vid, clip)

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs)

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':

    file = '/Users/lr/Desktop/VMR/data/dataset/processed_data'
    file2 = '/data/656146095/Danmu/processed_data'
    file3 = '/data/656146095/Danmu/Charades_v1_rgb'
    file4 = '/Users/lr/Desktop/VMR/data/TACoS/video'
    record = '/data/656146095/Danmu/charades_sta_train.txt'
    test_transforms = transforms.Compose([CenterCrop(224)])
    dataset = Danmus(file4, [1,1200],record_file=record, transforms= test_transforms,dataset='data3')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                                             pin_memory=True)

    for idx,x in enumerate(dataloader):
        pass
