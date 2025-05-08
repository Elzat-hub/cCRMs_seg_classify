import sys

sys.path.append('..')

import glob
import os
import random
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import nibabel as nib
import numpy as np


class EP(Dataset):
    def __init__(self, mode, dataset_path="../data"):
        self.mode = mode
        self.root = str(dataset_path)
        self.image_list = []
        self.seglabel_list = []
        self.classlabel_list = []

        self.train_path = self.root + '/EP/train/image'
        self.train_seglabel_path = self.root + '/EP/train/label'
        self.train_classlabel_path = self.root + '/train_classify_label.csv'
        self.test_path = self.root + '/EP/val/image'
        self.test_seglabel_path = self.root + '/EP/val/label'
        self.test_classlabel_path = self.root + '/test_classify_label.csv'

        list_train_image = sorted(glob.glob(os.path.join(self.train_path, '*.nii.gz')))
        list_train_seglabel = sorted(glob.glob(os.path.join(self.train_seglabel_path, '*.nii.gz')))
        list_train_classlabel = pd.read_csv(self.train_classlabel_path).copy()
        self.train_datanum = len(list_train_image)
        # self.train_datanum = len(list_train_classlabel)

        list_test_image = sorted(glob.glob(os.path.join(self.test_path, '*.nii.gz')))
        list_test_seglabel = sorted(glob.glob(os.path.join(self.test_seglabel_path, '*.nii.gz')))
        list_test_classlabel = pd.read_csv(self.test_classlabel_path).copy()
        self.val_datanum = len(list_test_image)

        # assert len(list_train_image) == len(list_train_seglabel) == len(list_train_classlabel)
        # assert len(list_test_image) == len(list_test_seglabel) == len(list_test_classlabel)
        assert len(list_train_image) == len(list_train_seglabel)
        assert len(list_test_image) == len(list_test_seglabel)

        split_idx = int(self.train_datanum * 0.8)


        random.seed(42)
        random.shuffle(list_train_image)
        random.seed(42)
        random.shuffle(list_train_seglabel)
        random.seed(42)
        random.shuffle(list_train_classlabel['image'].values)
        random.seed(42)
        random.shuffle(list_train_classlabel['label'].values)


        if self.mode == 'train':
            print('Single Tumor-EP Dataset for Training. Total data:', int(self.train_datanum * 0.8))
            self.image_list = list_train_image[:split_idx]
            self.seglabel_list = list_train_seglabel[:split_idx]
            # print(self.image_list)

        elif self.mode == 'val':
            print('Single Tumor-EP Dataset for Validating. Total data:', int(self.train_datanum * 0.2))
            self.image_list = list_train_image[split_idx:]
            self.seglabel_list = list_train_seglabel[split_idx:]
            # print(len(self.image_list),len(self.classlabel_list))

        elif self.mode == 'test':
            print('Single Tumor-EP Dataset for Test. Total data:', self.val_datanum)
            self.image_list = list_test_image
            self.seglabel_list = list_test_seglabel

    def __len__(self):
        return len(self.seglabel_list)

    def __getitem__(self, item):
        # 获取数据名
        image_name = os.path.basename(self.image_list[item])
        seg_lab_name = os.path.basename(self.seglabel_list[item])

        img = nib.load(self.image_list[item]).get_fdata()
        seg_lab = nib.load(self.seglabel_list[item]).get_fdata()

        # print("Item:", item)
        print(f"Accessing item {item} in mode {self.mode}")
        print(f"image_list length: {len(self.image_list)}")
        print(f"seglabel_list length: {len(self.seglabel_list)}")
        img = np.array(img).astype(np.float32)
        seg_lab = np.array(seg_lab).astype(np.float32)

        return torch.FloatTensor(img).unsqueeze(0), torch.FloatTensor(seg_lab).unsqueeze(0), image_name, seg_lab_name


if __name__ == '__main__':
    dataset = EP('test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
