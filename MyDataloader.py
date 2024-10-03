# -*- coding: utf-8 -*-

### cv2 read image
import torch
import pickle
from torch.utils import data
import numpy as np
import SimpleITK as sitk
from MyAugmentation import transform_img_lab
import warnings
import cv2

warnings.filterwarnings("ignore")


def read_file_from_txt(txt_path):
    files = []
    for line in open(txt_path, "r"):
        files.append(line.strip())
    return files


class Dataloader(data.Dataset):
    def __init__(self, args):
        super(Dataloader, self).__init__()
        self.image_file = read_file_from_txt(args.Image_Tr_txt)
        self.label_file = read_file_from_txt(args.Label_Tr_txt)
        self.shape = (args.ROI_shape, args.ROI_shape)
        self.args = args

    def __getitem__(self, index):
        image_path = self.image_file[index]
        label_path = self.label_file[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)  # [3,256,256]

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        y, x = label.shape
        # Normalization
        mean, std = np.load(self.args.data_dir + self.args.Tr_Meanstd_name)
        image = (image - mean) / std
        label = np.where(label > 0, 1, 0)

        center_y = np.random.randint(0, y - self.shape[0] + 1, 1, dtype=np.int16)[0]
        center_x = np.random.randint(0, x - self.shape[1] + 1, 1, dtype=np.int16)[0]
        image = image[:, center_y:self.shape[0] + center_y, center_x:self.shape[1] + center_x]
        label = label[center_y:self.shape[0] + center_y, center_x:self.shape[1] + center_x]

        label = label[np.newaxis, :, :]

        # Data Augmentation
        data_dict = transform_img_lab(image, label, self.args)
        image_trans = data_dict["image"]
        label_trans = data_dict["label"]
        if isinstance(image_trans, torch.Tensor):
            image_trans = image_trans.numpy()
        if isinstance(label_trans, torch.Tensor):
            label_trans = label_trans.numpy()

        return image_trans, label_trans

    def __len__(self):
        return len(self.image_file)


class Dataloader_test(data.Dataset):
    def __init__(self, args):
        super(Dataloader_test, self).__init__()
        self.image_file = read_file_from_txt(args.Image_Te_txt)
        self.label_file = read_file_from_txt(args.Label_Te_txt)
        self.shape = (args.ROI_shape, args.ROI_shape)
        self.args = args

    def __getitem__(self, index):
        image_path = self.image_file[index]
        label_path = self.label_file[index]
        name = image_path.split('/')[-1].split('.'[0])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)  # [3,256,256]

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        y, x = label.shape
        # Normalization
        mean, std = np.load(self.args.data_dir + self.args.Tr_Meanstd_name)
        image = (image - mean) / std
        label = np.where(label > 0, 1, 0)

        center_y = np.random.randint(0, y - self.shape[0] + 1, 1, dtype=np.int16)[0]
        center_x = np.random.randint(0, x - self.shape[1] + 1, 1, dtype=np.int16)[0]
        image = image[:, center_y:self.shape[0] + center_y, center_x:self.shape[1] + center_x]
        label = label[center_y:self.shape[0] + center_y, center_x:self.shape[1] + center_x]

        label = label[np.newaxis, :, :]

        return image, label, name

    def __len__(self):
        return len(self.image_file)
