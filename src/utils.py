# -*- coding: utf-8 -*-

from scipy import io
import numpy as np
import sklearn
import os
import PIL.Image as Image
from sklearn.metrics import confusion_matrix
import seaborn
datasets_name = {"indianpines":"IndianPines"}

current_dir = os.getcwd()

top_dir = os.path.join(os.getcwd(), "..")
datasets_dir = os.path.join(top_dir, "Datasets")

# about dataset op
def load_dataset(dataset_name, hyperparams):
    folder = "Datasets"
    if dataset_name == "IndianPines":
        img_path = os.path.join(current_dir, "..", folder, dataset_name, "Indian_pines_corrected.mat")
        gt_path = os.path.join(current_dir, "..", folder, dataset_name, "Indian_pines_gt.mat")
        # uint16
        hsi_img = io.loadmat(img_path)["indian_pines_corrected"]
        gt = io.loadmat(gt_path)["indian_pines_gt"]
        labels = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]
        ignored_labels = [0]
        rgb_bands = (43, 21, 11)
    else:
        pass
    
    palette = get_palette(len(labels))
    hyperparams["labels"] = labels
    hyperparams["n_classes"] = len(labels)
    hyperparams["ignored_labels"] = ignored_labels
    
    
    # real rgb image
    real_rgb_path = os.path.join(datasets_dir, dataset_name, "real_rgb.png")
    if os.path.exists(real_rgb_path):
        print("--- real_rgb.png already exists")
    else:
        real_rgb_IMG = hsi_to_rgb(hsi_img, rgb_bands = rgb_bands)
        real_rgb_IMG.save(real_rgb_path)
    # gt rgb image
    gt_rgb_path = os.path.join(datasets_dir, dataset_name, "gt_rgb.png")
    if os.path.exists(gt_rgb_path):
        print("--- gt_rgb.png already exists")
    else:
        gt_rgb_IMG = label_to_color(gt, palette, ignored_labels)
        gt_rgb_IMG.save(gt_rgb_path)
    
    return hsi_img, gt, palette


def split_train_test(gt, trainrate):
    """
    划分训练集和测试集
    :param gt:  height * width 2D int labels
    :param trainrate:训练集比例
    :return: train_gt, test_gt
    """
    # (2, n) 找到所有非零元素的索引，第一行为行索引，第二行为列索引
    indices = np.nonzero(gt)
    # 获得元组(r,c)格式的坐标
    X = list(zip(*indices))
    y = gt[indices].ravel()  # 获得对应坐标的真值
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    # random sample
    # 使用train_test_split函数随机划分训练集和测试集，保持每个类别的比例相同。
    train_indices, test_indices = sklearn.model_selection.train_test_split(
            X, train_size = trainrate, random_state = 0,
            stratify = y)
    # [(r0,c0)(r1,c1)(..)...] to [[r0, r1, ...], [c0, c1, ...]]
    train_indices = [list(t) for t in zip(*train_indices)]
    test_indices = [list(t) for t in zip(*test_indices)]
    # 获得训练集和测试集的真值
    train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
    test_gt[tuple(test_indices)] = gt[tuple(test_indices)]
    return train_gt, test_gt

def build_dataset(img, gt):
    """
    构建数据:wq
    :集
    :param img: height * width * bands 3D float image
    :param gt: height * width 2D int labels
    :return: X_train, y_train
    """
    samples = []
    labels = []
    for label in np.unique(gt):
        if label == 0:
            continue
        # 找到所有真值==label的像素，格式为：
        # ([r0, r1, ...],[c0, c1, ...])
        indices = np.nonzero(gt == label)
        samples += list(img[indices])
        labels += len(indices[0]) * [label]
    X_train = np.asarray(samples)
    y_train = np.asarray(labels)
    return X_train, y_train

def build_dataset_cnn2d(img, gt, patch_size):
    """
    构建数据:wq
    :集
    :param img: height * width * bands 3D float image
    :param gt: height * width 2D int labels
    :return: X_train, y_train
    """
    samples = []
    labels = []
    for label in np.unique(gt):
        if label == 0:
            continue
        # 找到所有真值==label的像素，格式为：
        # ([r0, r1, ...],[c0, c1, ...])
        indices = np.nonzero(gt == label)
        samples += get_window(img, gt.shape[0],indices, patch_size)
        labels += len(indices[0]) * [label]
    X_train = np.asarray(samples)
    y_train = np.asarray(labels)
    return X_train, y_train

def get_window(img, img_length, indices ,patch_size):
    X_get = []
    for posX,posY in zip(indices[0],indices[1]):
        # posX,posY = int(posX),int(posY)
        X = np.zeros((patch_size,patch_size,img.shape[2]),dtype = np.float32)
        if (posX - patch_size//2 >-1):
            left = posX - patch_size//2
        else:
            left = 0
        if(posX + patch_size//2 < img_length):
            right = posX + patch_size//2
        else:
            right = img_length
        if(posY - patch_size//2 >-1):
            bottom = posY - patch_size//2
        else:
            bottom = 0
        if(posY + patch_size//2 < img_length):
            top = posY + patch_size//2 
        else:
            top = img_length
        X[0:right-left,0:top-bottom] =  img[left:right,bottom:top]
        X_get.append(X)
    return X_get



def hsi_to_rgb(img, rgb_bands = None):
    # 原图
    r = img[:, :, rgb_bands[0]] * 1.55 // 80
    g = img[:, :, rgb_bands[1]] * 1.55 // 80
    b = img[:, :, rgb_bands[2]] * 1.55 // 80

    r = r.astype(np.int32)
    b = b.astype(np.int32)
    g = g.astype(np.int32)
    rgb_IMG = Image.new("RGB", (145, 145))
    for i in range(145):
        for j in range(145):
            rgb_IMG.putpixel((i, j), (r[i, j], g[i, j], b[i, j]))
    return rgb_IMG

def label_to_color(arr_2d, palette = None, ignored_labels =[0]):
    """从标签转换为颜色图像。

    Args:
        arr_2d: 2D numpy array，标签图像
        palette: dict，标签到颜色的映射
    Returns:
        arr_3d: 3D numpy array，颜色图像
    """
    if palette is None:
        palette = {0: (0, 0, 0)}
        for k, color in enumerate(seaborn.color_palette("hls", 21)):
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype = "uint8"))

    # 根据调色板创建颜色图像
    arr_rgb = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype = "uint8")
    for c, i in palette.items():
        m = arr_2d == c
        arr_rgb[m] = i

    return arr_rgb

def save_pred(
        pred_img,
        palette, # 色彩板
        res_folder = "result", # 相对主目录的路径
        name = "",
        hyperparams = {},
        accuracy = 0,):

    abs_res_folder = os.path.join(os.getcwd(), "..", res_folder, "{preprocess}_{model}".format(**hyperparams) )
    if not os.path.exists(abs_res_folder):
        os.makedirs(abs_res_folder)
    
    color_pred = label_to_color(pred_img, palette)
    color_pred_IMG = Image.fromarray(color_pred)
    color_pred_IMG.save(os.path.join(abs_res_folder, name + "acy%.2f.png"%accuracy))

def get_vector_mask(vector_gt, ignored_labels):
    # vector_gt = gt.reshape(-1)
    vector_mask = np.zeros(vector_gt.shape, dtype = np.bool)
    for ig_label in ignored_labels:
        vector_mask[vector_gt == ig_label] = 1
    return vector_mask



def metrics(prediction, 
            target, 
            ignored_labels = [0], 
            n_classes = 0,
            ):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = ~ get_vector_mask(target, ignored_labels)
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    conf_matrix = confusion_matrix(target, prediction, labels = range(n_classes))

    results["confusion_matrix"] = conf_matrix

    # Compute global accuracy
    total = np.sum(conf_matrix)
    accuracy = sum([conf_matrix[x][x] for x in range(len(conf_matrix))])
    accuracy *= 100 / float(total)

    results["accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(conf_matrix))
    for i in range(len(conf_matrix)):
        try:
            F1 = 2. * conf_matrix[i, i] / (np.sum(conf_matrix[i, :]) + np.sum(conf_matrix[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["f1_scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(conf_matrix) / float(total)
    pe = np.sum(np.sum(conf_matrix, axis = 0) * np.sum(conf_matrix, axis = 1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["kappa"] = kappa

    return results


def show_results(run_results, hyperparams):
    """Print results of a run."""
    print("\n\
* Dataset {dataset}, Training rate {training_rate}, \n\
* Preprocess {preprocess}, N_Bands {n_bands}, \n\
* Model {model}, N_Runs {n_runs}, Patch_Size {patch_size}, Batch_Size {batch_size}".format(**hyperparams))
    
    print("Confusion matrix:")
    for i in range(run_results["confusion_matrix"].shape[0]):
        for j in range(run_results["confusion_matrix"].shape[1]):
            # 设置对齐
            print("{:4d}".format(run_results["confusion_matrix"][i][j]), end = " ")
        print()

    print("F1 scores:")
    for label, score in zip(hyperparams["labels"], run_results["f1_scores"]):
        print(" - {:30}: {:.2f}".format(label, score))

    print("Kappa: {:.2f}".format(run_results["kappa"]))
    print("Global accuracy: {:.2f}".format(run_results["accuracy"]))



def get_palette(n_classes):
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(seaborn.color_palette("hls", n_classes - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype = "uint8"))