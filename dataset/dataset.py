from PIL import Image
from torch.nn.functional import batch_norm
from torch.utils.data import Dataset
import torch
from torchvision import transforms as T
import os
import torchvision.transforms.functional as TF
import random
import numpy as np
from tqdm import tqdm
import albumentations as A
import cv2
import shutil
from albumentations import (
    HorizontalFlip, Perspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    MotionBlur, MedianBlur,
    Sharpen, Emboss, RandomBrightnessContrast, Flip, OneOf, Compose, ReplayCompose
)
import soundfile as sf
from torch import Tensor
import dataset_collate


trans = {300:T.Compose([T.Resize(300), T.ToTensor()]),
         128:T.Compose([T.Resize((128, 128)), T.ToTensor()]),
         299:T.Compose([T.ToTensor(), T.Resize(299)]),
         256:T.Compose([T.ToTensor(), T.Resize((256, 256))]),
         224:T.Compose([T.ToTensor(), T.Resize((224, 224))]),
         192:T.Compose([T.ToTensor(),T.Resize((192, 192))])}


class MyDataset_multiframe(Dataset):
    def __init__(self, image_size, type, txt_path, num_frame):
        assert image_size in trans.keys()
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            #print(line)
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.trans = trans[image_size]
        self.image_size = image_size
        self.num_frame = num_frame
        self.phase = type

    def __getitem__(self, index):
        data=torch.zeros((self.num_frame, 3, self.image_size, self.image_size))
        fn, label = self.imgs[index]
        filename = fn
        name = os.path.split(temp)[1]
        index = int(os.path.split(fn)[1][:-4])
        temp1 = ''

        for i in range(self.num_frame):

            fn = temp + '/' + str(index + i).zfill(4) + '.png'

            if not os.path.exists(fn):
                fn = temp1
            try:
                img = Image.open(fn).convert('RGB')
                img = self.trans(img)
                data[i, :, :, :] = img
                temp1 = fn
            except:
                print('........................')
                print(filename)

        label = label
        data = data.view(-1, self.image_size, self.image_size)

        return name, data, label

    def __len__(self):
        return len(self.imgs)

