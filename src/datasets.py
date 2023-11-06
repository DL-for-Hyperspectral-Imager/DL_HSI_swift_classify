# -*- coding: utf-8 -*-


from torch.utils.data.dataset import Dataset
import numpy as np
import math


class Mydatasets(Dataset):
    def __init__(self, img_X, img_y):
        self.X = img_X
        self.y = img_y
        assert len(img_X) == len(img_y)  # 查看x和y的数量是否相同
        self.len = len(self.X)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        return X, y


class CNNDatasets(Dataset):
    def __init__(self, img_X, img_y, patch_size):
        img_length = int(math.sqrt(len(img_X)))
        self.len = img_length**2
        self.X = img_X[: img_length**2, :].reshape(
            img_length, img_length, img_X.shape[1]
        )
        self.y = img_y[: img_length**2].reshape(img_length, img_length)
        self.phs = patch_size
        assert len(img_X) == len(img_y)  ##查看x和y的数量是否相同
        self.indices = np.array(
            [[x, y] for x in range(img_length) for y in range(img_length)]
        )
        self.img_length = img_length

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        posX, posY = self.indices[idx]
        X = np.zeros((self.phs, self.phs, self.X.shape[2]), dtype=np.float32)
        if posX - self.phs // 2 > -1:
            left = posX - self.phs // 2
        else:
            left = 0
        if posX + self.phs // 2 < self.img_length:
            right = posX + self.phs // 2
        else:
            right = self.img_length
        if posY - self.phs // 2 > -1:
            bottom = posY - self.phs // 2
        else:
            bottom = 0
        if posY + self.phs // 2 < self.img_length:
            top = posY + self.phs // 2
        else:
            top = self.img_length
        X[0 : right - left, 0 : top - bottom] = self.X[left:right, bottom:top]
        y = self.y[posX, posY]
        return np.array(X), np.array(y)
