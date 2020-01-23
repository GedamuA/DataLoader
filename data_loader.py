import os, sys
from os import listdir
from os.path import join, isfile

import numpy as np
from PIL import Image
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pdb
import glob

dd = pdb.set_trace
views = ["0", "2", "4", "6", "4", "6", "0", "2"]
image_dir = "/home/gede/VcGAN/Dataset/Train_image/"
# ForTrain = "/home/gede/VcGAN/Dataset/ForTrain/"
image = "image not found"


def read_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((128, 128), Image.ANTIALIAS)
    return img


def read_for_train_img(imagName):
    global image
    for img in glob.glob(image_dir + imagName + "*.*"):
        image = img
        if img is not None:
            break
    return image


def get_test_img(img_path):
    tmp = random.randint(0, 7)
    tmp2 = random.randint(1, 7)
    view2 = int(views[tmp])
    token = img_path.split("/")
    img2 = token[6][0:5] + str(view2) + '_d'
    img2 = read_for_train_img(img2)
    img2 = Image.open(img2).convert('RGB')
    img2 = img2.resize((128, 128), Image.ANTIALIAS)
    return view2, img2


class ImageList(data.Dataset):
    def __init__(self, list_file, transform=None, is_train=True,
                 img_shape=None):
        # img_list = [line.rstrip('\n') for line in open(list_file)]
        if img_shape is None:
            img_shape = [128, 128]
        print('total %d images' % len(list_file))

        self.img_list = list_file
        self.transform = transform
        self.is_train = is_train
        self.img_shape = img_shape
        self.transform_img = transforms.Compose([self.transform])

    def __getitem__(self, index):
        # img_path:"/home/gede/VcGAN/Dataset/images/a00_v2_p001_c1.png"
        img1_path = self.img_list[index]
        token = img1_path.split('/')
        action = int(token[6][1:3])
        view1 = int(token[6][5])
        img1 = read_img(img1_path)
        view2, img2 = get_test_img(img1_path)
        if self.transform_img is not None:
            img1 = self.transform_img(img1)  # [0,1], c x h x w
            img2 = self.transform_img(img2)

        return view1, view2, img1, img2

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    image_dir = "/home/gede/VcGAN/Dataset/Train_image/"

    imgFiles_train = [
        join(image_dir, fn)  # Create full paths to images
        for fn in listdir(image_dir)  # For each item in the image folder
        if isfile(join(image_dir, fn))  # If the item is indeed a file
           and fn.lower().endswith(('.png', '.jpg'))
        # Which ends with an image suffix (can add more types here if needed)
    ]



