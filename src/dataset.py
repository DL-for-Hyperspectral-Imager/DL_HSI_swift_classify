# -*- coding: utf-8 -*-

import os

from scipy.io import loadmat
from sklearn.model_selection import train_test_split

import numpy as np

import torch
from torch.utils.data.dataset import Dataset


from dataset_info import DATASET_INFO
from utils import get_palette, hsi_to_rgbIMG, label_to_color_IMG

current_dir = os.getcwd()

top_dir = os.path.join(os.getcwd(), "..")
datasets_dir = os.path.join(top_dir, "Datasets")


class ImgDataset(Dataset):
    def __init__(self, img_X, img_y):
        assert len(img_X) == len(img_y)  # 查看x和y的数量是否相同
        self.X = torch.from_numpy(img_X)
        self.y = torch.from_numpy(img_y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# about dataset op
def load_dataset(ds_name, dtype=np.float32) -> (np.ndarray, np.ndarray, dict):
    """

    Returns:
        hsi_img: numpy.ndarray, (height, width, bands), uint16
                spectral intensity value
        gt:      numpy.ndarray, (height, width),        uint8
                label for each pixel
        palette: dict({label: (r, g, b), ...})
                palette for drawing
    """

    print(f"* Loading dataset {ds_name}...")

    folder = "Datasets"

    if ds_name not in DATASET_INFO.keys():
        raise Exception("Unknown Dataset Name!")

    ds_dir = os.path.join(current_dir, "..", folder, ds_name)
    img_path = os.path.join(ds_dir, DATASET_INFO[ds_name]["img"])
    gt_path = os.path.join(ds_dir, DATASET_INFO[ds_name]["gt"])

    hsi_img: np.ndarray
    gt: np.ndarray

    if ds_name == "IndianPines":
        # uint16
        hsi_img = loadmat(img_path)["indian_pines_corrected"]
        hsi_img = hsi_img.astype(dtype)
        gt = loadmat(gt_path)["indian_pines_gt"]

    elif ds_name == "XiongAn":
        from osgeo import gdal

        hsi_img = gdal.Open(img_path).ReadAsArray()
        hsi_img = hsi_img.astype(dtype)
        hsi_img = hsi_img.transpose(1, 2, 0)

        gt = gdal.Open(gt_path).ReadAsArray()
        hsi_img, gt = ds_sample(hsi_img, gt, gt.shape, (200, 500))

    else:
        raise Exception("Unknown Dataset Name!")

    labels = DATASET_INFO[ds_name]["labels"]
    ignored_labels = DATASET_INFO[ds_name]["ignored"]
    rgb_bands = DATASET_INFO[ds_name]["rgb"]

    save_ds_image(ds_name, hsi_img, gt, labels, rgb_bands)

    print(f"* Dataset {ds_name} loaded!")
    print("--- Original dataset shape:", hsi_img.shape)

    return hsi_img, gt, labels, ignored_labels


def ds_sample(hsi_img, gt, inshape: tuple, outshape: tuple):
    x_step = inshape[0] / outshape[0]
    y_step = inshape[1] / outshape[1]

    x_indices = np.arange(0, inshape[0], x_step, dtype=int)
    y_incices = np.arange(0, inshape[1], y_step, dtype=int)

    hsi_img = hsi_img[
        np.ix_(
            x_indices,
            y_incices,
        )
    ]
    gt = gt[
        np.ix_(
            x_indices,
            y_incices,
        )
    ]
    return hsi_img, gt
    # x = 500
    # y = 1000
    # x_delta = 200
    # y_delta = 1000
    # hsi_img = hsi_img[x : x + x_delta, y : y + y_delta, :]
    # gt = gt[x : x + x_delta, y : y + y_delta]


def split_train_test(gt, trainrate):
    """
    划分训练集和测试集
    :param gt:  height * width 2D int labels
    :param trainrate: 训练集比例
    :return:
        train_gt: ndarray, (h, w)
            if train_gt[i][j] == 0, means this pixel do not use
        test_gt:  ndarray, (h, w)
            if test_gt[i][j] == 0, means this pixel do not use
    """
    print(f"* Splitting train and test set..., training rate: {trainrate}")
    # (2, n) 找到所有非零元素的索引，第一行为行索引，第二行为列索引
    indices = np.nonzero(gt)
    # 获得元组(r,c)格式的坐标
    X = list(zip(*indices))
    y = gt[indices].ravel()  # 获得对应坐标的真值
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    # random sample
    # 使用train_test_split函数随机划分训练集和测试集，保持每个类别的比例相同。
    train_indices, test_indices = train_test_split(
        X, train_size=trainrate, random_state=0, stratify=y
    )
    # [(r0,c0)(r1,c1)(..)...] to [[r0, r1, ...], [c0, c1, ...]]
    train_indices = [list(t) for t in zip(*train_indices)]
    test_indices = [list(t) for t in zip(*test_indices)]
    # 获得训练集和测试集的真值
    train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
    test_gt[tuple(test_indices)] = gt[tuple(test_indices)]
    return train_gt, test_gt


# def split(gt, trainrate):


def build_Xy(img, gt, is_win=False, patch_size=None):
    """
    构建数据集:
    :param img: height * width * bands 3D float image
    :param gt: height * width 2D int labels
    :return:
        X_train: (pixel_nums, features)
        y_train: (pixel_nums,)
    """
    samples = []
    labels = []
    labels_indices = {}
    for label in np.unique(gt):
        if label == 0:
            continue
        # 找到所有真值==label的像素，格式为：
        # ([r0, r1, ...],[c0, c1, ...])
        indices = np.nonzero(gt == label)
        labels_indices[label] = indices
        if is_win:
            samples += get_windows(img, indices, patch_size)
        else:
            samples += list(img[indices])

        labels += len(indices[0]) * [label]
    X_train = np.asarray(samples)
    y_train = np.asarray(labels)
    return X_train, y_train, labels_indices


def get_windows(img: np.ndarray, indices, patch_size: int):
    windows = []
    psz2 = patch_size // 2
    for row, col in zip(indices[0], indices[1]):
        window = np.zeros((patch_size, patch_size, img.shape[2]), dtype=np.float32)
        r1 = max(row - psz2, 0)
        r2 = min(row + psz2, img.shape[0])

        c1 = max(col - psz2, 0)
        c2 = min(col + psz2, img.shape[1])

        window[0 : r2 - r1, 0 : c2 - c1] = img[r1:r2, c1:c2]
        windows.append(window)
        # print(r1, r2, c1, c2)
        # break
    return windows


def save_ds_image(dataset_name, hsi_img, gt, labels, rgb_bands):
    palette = get_palette(len(labels))

    # real rgb image
    real_rgb_path = os.path.join(datasets_dir, dataset_name, "real_rgb.png")

    if os.path.exists(real_rgb_path):
        print("--- real_rgb.png already exists")
        real_rgb_IMG = hsi_to_rgbIMG(hsi_img, rgb_bands)
        real_rgb_IMG.save(real_rgb_path)
    else:
        real_rgb_IMG = hsi_to_rgbIMG(hsi_img, rgb_bands)
        real_rgb_IMG.save(real_rgb_path)

    # gt rgb image
    gt_rgb_path = os.path.join(datasets_dir, dataset_name, "gt_rgb.png")
    if os.path.exists(gt_rgb_path):
        print("--- gt_rgb.png already exists")
        gt_rgb_IMG = label_to_color_IMG(gt, palette)
        gt_rgb_IMG.save(gt_rgb_path)
    else:
        gt_rgb_IMG = label_to_color_IMG(gt, palette)
        gt_rgb_IMG.save(gt_rgb_path)
