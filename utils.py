# -*- coding: utf-8 -*-

from scipy import io
import numpy as np
import sklearn
import os
import PIL.Image as Image
from sklearn.metrics import confusion_matrix

current_dir = os.getcwd()


def load_dataset(dataset_name):
    folder = "Datasets"
    if dataset_name == "IndianPines":
        img_path = os.path.join(current_dir, folder, dataset_name, "Indian_pines_corrected.mat")
        gt_path = os.path.join(current_dir, folder, dataset_name, "Indian_pines_gt.mat")
        # uint16
        img = io.loadmat(img_path)["indian_pines_corrected"]
        gt = io.loadmat(gt_path)["indian_pines_gt"]
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]
    else:
        pass

    # # 数据转换
    # img = np.transpose(img, (2, 0, 1))
    # gt = np.asarray(gt, dtype=np.int32)
    # gt = np.transpose(gt, (1, 0))

    return img, gt, label_values


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
    train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=trainrate, random_state=0,
                                                                           stratify=y)
    # [(r0,c0)(r1,c1)(..)...] to [[r0, r1, ...], [c0, c1, ...]]
    train_indices = [list(t) for t in zip(*train_indices)]
    test_indices = [list(t) for t in zip(*test_indices)]
    # 获得训练集和测试集的真值
    train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
    test_gt[tuple(test_indices)] = gt[tuple(test_indices)]
    return train_gt, test_gt


def build_set(img, gt):
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


def hsi2rgb(img):
    # 原图
    b = img[:, :, 3] * 1.55 // 80
    g = img[:, :, 13] * 1.55 // 80
    r = img[:, :, 28] * 1.55 // 80
    r = r.astype(np.int32)
    b = b.astype(np.int32)
    g = g.astype(np.int32)
    rgb_IMG = Image.new("RGB", (145, 145))
    for i in range(145):
        for j in range(145):
            rgb_IMG.putpixel((i, j), (r[i, j], g[i, j], b[i, j]))
    return rgb_IMG


def visualize(hsi_img, gt, pred_img):
    rgb_IMG = hsi2rgb(hsi_img)

    # 颜色
    color = [(255, 255, 255), (255, 0, 0), (255, 128, 0), (255, 255, 0), (128, 255, 0), \
             (0, 255, 0), (0, 255, 255), (0, 128, 255), (0, 0, 255), (127, 0, 255), (255, 0, 255), \
             (153, 0, 0), (0, 102, 51), (153, 153, 255), (255, 204, 204), (255, 255, 204), (0, 0, 0)]

    # 原标签对应图
    color_gt_IMG = Image.new("RGB", (145, 145))
    for i in range(145):
        for j in range(145):
            color_gt_IMG.putpixel((i, j), color[gt[i, j]])

    # 预测标签对应图
    color_pred_gt_IMG = Image.new("RGB", (145, 145))
    for i in range(145):
        for j in range(145):
            color_pred_gt_IMG.putpixel((i, j), color[pred_img[i, j]])

    resdir = os.path.join(current_dir, "result")
    os.makedirs(resdir, exist_ok=True)

    rgb_IMG.save(os.path.join(resdir, "rgb.png"))
    color_gt_IMG.save(os.path.join(resdir, "color_gt.png"))
    color_pred_gt_IMG.save(os.path.join(resdir, "color_pred_gt.png"))

    # rgb_IMG.show()
    # color_gt_IMG.show()
    # color_pred_gt_IMG.show()


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool_)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    # target = target[ignored_mask] -1
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(target, prediction, labels=range(n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1 scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
         float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    return results


def show_results(args, run_results, label_values):
    """Print results of a run."""
    print("Results for {} on {} with {} and {}% sample rate:".format(
        args.model, args.dataset, args.preprocess, args.training_sample * 100))

    print("Confusion matrix:")
    for i in range(run_results["Confusion matrix"].shape[0]):
        for j in range(run_results["Confusion matrix"].shape[1]):
            # 设置对齐
            print("{:4d}".format(run_results["Confusion matrix"][i][j]), end=" ")
        print()

    print("Global accuracy: {:.2f}".format(run_results["Accuracy"]))
    print("F1 scores:")
    for label, score in zip(label_values, run_results["F1 scores"]):
        print("  {}: {:.2f}".format(label, score))

    print("Kappa: {:.2f}".format(run_results["Kappa"]))
