import glob
import os

from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from torchvision import transforms
import torch.nn.functional as F


class Preprocessor:
    def __init__(self, train_path='images_train'):
        files = sorted(glob.glob(os.path.join(train_path, '*.png')))

        self.grouped_files = self.get_grp(files)

        label = pd.read_csv('data/y_train.csv')

        self.enc = LabelEncoder()
        self.enc.fit(label['cell_line'])

        self.label_id = self.enc.transform(label['cell_line'])

        (self.X_train, self.y_train), (self.X_test, self.y_test) = self.split_data()

    def split_data(self):
        x_train, x_test, y_train, y_test = train_test_split(self.grouped_files, self.label_id,
                                                            stratify=self.label_id, test_size=1000)
        return (x_train, y_train), (x_test, y_test)

    def get_grp(self, files):
        triplet = []
        groups = []
        for i, file in enumerate(files):
            triplet.append(file)
            if (i + 1) % 3 == 0:
                groups.append(triplet)
                triplet = []
        return groups

    def get_img(self, grp):
        dict_image = {}
        for i, ims in enumerate(grp):
            im_1 = np.array(Image.open(ims[0]))
            im_2 = np.array(Image.open(ims[1]))
            im_3 = np.array(Image.open(ims[2]))
            dict_image[i] = np.stack((im_1, im_2, im_3))
        return dict_image

    def get_train(self):
        x = self.get_img(self.X_train)
        return x, self.y_train

    def get_val(self):
        x = self.get_img(self.X_test)
        return x, self.y_test

    def get_test(self, test_path='images_test'):
        files = sorted(glob.glob(os.path.join(test_path, '*.png')))
        groups = self.get_grp(files)
        return self.get_img(groups)


class TrainDataset(Dataset):
    def __init__(self, data):
        x, y = data
        self.x = x
        self.y = torch.LongTensor(y)

        self.trf = transforms.Normalize(mean=[32.93115911135246, 33.631578515138344, 37.49410179366305],
                                        std=[56.50129412862437, 42.07109710504794, 40.591463344084985],
                                        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x_ten = torch.FloatTensor(self.x[item])
        return self.trf(x_ten), self.y[item]


class TestDataset(Dataset):
    def __init__(self, data):
        self.x = data

        self.trf = transforms.Normalize(mean=[32.93115911135246, 33.631578515138344, 37.49410179366305],
                                        std=[56.50129412862437, 42.07109710504794, 40.591463344084985],
                                        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x_ten = torch.FloatTensor(self.x[item])
        return self.trf(x_ten)
