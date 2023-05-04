from scipy import io
import numpy as np
import sklearn
import os
import PIL.Image as Image

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

    rgb_IMG.save(os.path.join(resdir,"rgb.png"))
    color_gt_IMG.save(os.path.join(resdir,"color_gt.png"))
    color_pred_gt_IMG.save(os.path.join(resdir,"color_pred_gt.png"))

    # rgb_IMG.show()
    # color_gt_IMG.show()
    # color_pred_gt_IMG.show()
