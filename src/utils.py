# -*- coding: utf-8 -*-

from scipy import io
import numpy as np
import os
import PIL.Image as Image
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import seaborn


current_dir = os.getcwd()

top_dir = os.path.join(os.getcwd(), "..")
datasets_dir = os.path.join(top_dir, "Datasets")


def get_palette(n_classes):
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(seaborn.color_palette("hls", n_classes - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
    return palette


def get_img_pred(y_test_all_pred, labels_indices: dict, shape: tuple):
    pred_img = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    cnt = 0
    for label, indices in labels_indices.items():
        for r, c in zip(*indices):
            pred_img[r][c] = y_test_all_pred[cnt].item()
            cnt += 1
    return pred_img


def hsi_to_rgbIMG(img, rgb_bands=None):
    arr_rgb = img[:, :, rgb_bands]
    arr_rgb = arr_rgb / arr_rgb.max() * 256
    arr_rgb = arr_rgb.astype(np.uint8)

    rgb_IMG = Image.fromarray(arr_rgb, mode="RGB")
    return rgb_IMG


def label_to_color_IMG(arr_2d, palette):
    """从标签转换为颜色图像。

    Args:
        arr_2d: 2D numpy array, 标签图像
        palette: dict, 标签到颜色的映射
    Returns:
        arr_3d: 3D numpy array，颜色图像
    """

    # 根据调色板创建颜色图像
    arr_rgb = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype="uint8")
    for c, i in palette.items():
        m = arr_2d == c
        arr_rgb[m] = i
    color_IMG = Image.fromarray(arr_rgb, mode="RGB")
    return color_IMG


def save_pred(
    pred_img,
    n_classes,
    res_path="result",  # 相对主目录的路径
    name="",
    **kwargs,
):
    palette = kwargs.get("palette", get_palette(n_classes))

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    color_pred_IMG = label_to_color_IMG(pred_img, palette)
    color_pred_IMG.save(os.path.join(res_path, name))


def get_vector_mask(vector_gt, ignored_labels):
    # vector_gt = gt.reshape(-1)
    vector_mask = np.zeros(vector_gt.shape, dtype=np.bool_)
    for ig_label in ignored_labels:
        vector_mask[vector_gt == ig_label] = 1
    return vector_mask


def metrics_beta(y_true, y_pred, n_classes):
    results = {}

    conf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=range(n_classes))

    accuracy = metrics.accuracy_score(y_true, y_pred)

    f1score = metrics.f1_score(y_true, y_pred, average="weighted")

    kappa = metrics.cohen_kappa_score(y_true, y_pred)

    results["confusion_matrix"] = conf_matrix
    results["accuracy"] = accuracy
    results["f1score"] = f1score
    results["kappa"] = kappa
    return results


def metrics_old(
    prediction,
    target,
    ignored_labels=[0],
    n_classes=0,
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
    ignored_mask = ~get_vector_mask(target, ignored_labels)
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    conf_matrix = confusion_matrix(target, prediction, labels=range(n_classes))

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
            F1 = (
                2.0
                * conf_matrix[i, i]
                / (np.sum(conf_matrix[i, :]) + np.sum(conf_matrix[:, i]))
            )
        except ZeroDivisionError:
            F1 = 0.0
        F1scores[i] = F1

    results["f1_scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(conf_matrix) / float(total)
    pe = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / float(
        total * total
    )
    kappa = (pa - pe) / (1 - pe)
    results["kappa"] = kappa

    return results


def show_result(result, hparams):
    """Print results of a run."""

    print(
        "\
    * Dataset {dataset}, Training rate {train_rate}, \n\
    * Preprocess {preprocess}, N_Bands {t_bands}, \n\
    * Model {model}, N_Runs {n_runs}, Patch_Size {patch_size}, Batch_Size {batch_size}".format(
            **hparams
        )
    )

    print("Confusion matrix:")
    conf_matrix = result["confusion_matrix"]
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            # 设置对齐
            print(f"{conf_matrix[i][j]:4d}", end=" ")
        print()

    print(f"f1score        : {result['f1score']:.2f} (weighted)")
    print(f"Kappa          : {result['kappa']:.2f}")
    print(f"Global accuracy: {result['accuracy']:.2f}")
    print(f"Preprocess time: {result['preprocess_time']:.2f}s")
    print(f"Train time     : {result['train_time']:.2f}s")
    print(f"Predict time   : {result['predict_time']:.2f}s")


import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_loss_acc(loss, acc):
    assert len(loss) == len(acc)
    epoch = [i for i in range(len(loss))]
    fig, ax = plt.subplots()
    fig: Figure
    ax: Axes
    ax.plot(epoch, loss, label="loss")
    ax.plot(epoch, acc, label="acc")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss/acc")

    ax.set_ylim(0, 1)

    ax.legend()
    plt.show()
