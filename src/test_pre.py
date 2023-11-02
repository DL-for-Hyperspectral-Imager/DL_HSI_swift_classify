import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import seaborn as sns
import os

from src.utils import load_dataset
from src.preprocess import preprocess
import time
test_pre_fpath = os.path.join(os.getcwd(), "..", "temp", "test_pre")

pre_list = ['pca']

def measure_pre_time(hyperparams = {}):
    hsi_img, gt, palette = load_dataset("IndianPines", hyperparams)
    for pre in pre_list:
        time_list = []
        hyperparams["preprocess"] = pre
        bands_list = np.arange(150, 201, 1)
        bands_list = bands_list.astype(int)
        if pre == "lda":
            bands_list = list(np.arange(2, 16))
        for n_bands in bands_list:
            hyperparams["n_bands"] = n_bands
            start = time.time()
            img = preprocess(
                    hsi_img = hsi_img, gt = gt,
                    preprocess_name = hyperparams["preprocess"],
                    n_bands = hyperparams["n_bands"])
            end = time.time()
            time_cost = end - start
            time_list.append(time_cost)
            print("pre: {pre}, n_bands: {n_bands}, time_cost: {time_cost}".format(
                pre = pre, n_bands = n_bands, time_cost = time_cost))
        plt.figure(figsize = (8, 6))
        plt.plot(bands_list, time_list, "oy-", label = pre)

        plt.xlabel("Bands", fontsize = 14)
        plt.ylabel("Time (s)", fontsize = 14)
        plt.legend()
        name = "{preprocess}_{n_bands}".format(**hyperparams)
        plt.savefig(os.path.join(test_pre_fpath, name+ "_pre_time.png"))
        plt.clf()
        plt.close()
    return end - start


def test_pre(hyperparams = {}):
    hsi_img, gt, palette = load_dataset(hyperparams["dataset"], hyperparams)
    img = preprocess(
        hsi_img = hsi_img, gt = gt,
        preprocess_name = hyperparams["preprocess"],
        n_bands = hyperparams["n_bands"])
    img_vec = img.reshape(-1, img.shape[2])
    gt_vec = gt.reshape(-1)
    draw_scatter_2d(img_vec, gt_vec, hyperparams)
    draw_scatter_3d(img_vec, gt_vec, hyperparams)
    draw_corr_matrix(img_vec, hyperparams)

def draw_corr_matrix(img_vec, hyperparams = {}):
    corr_matrix = np.corrcoef(img_vec, rowvar = False)
    plt.figure(figsize = (10, 10))
    sns.heatmap(corr_matrix, square = True, cmap = 'coolwarm')
    name = "{preprocess}_{n_bands}".format(**hyperparams)
    plt.savefig(os.path.join(test_pre_fpath, name + "_corr_matrix.png"))
    plt.show()
def draw_scatter_3d(img_vec, gt_vec, hyperparams = {}):
    gt_vec = np.where(gt_vec == 0, np.nan, gt_vec)
    cmap = plt.get_cmap('tab20', 16)
    # cmap = plt.get_cmap('viridis', 16)
    fig = plt.figure(figsize = (15, 15))
    gs = gridspec.GridSpec(2, 2, width_ratios = [10, 1], height_ratios = [10, 1])
    ax = plt.subplot(gs[0], projection = '3d')
    sc = ax.scatter(img_vec[:, 0], img_vec[:, 1], img_vec[:, 2], c = gt_vec, cmap = cmap)
    ax.view_init(20, 70)
    ax2 = plt.subplot(gs[0, 1])
    plt.colorbar(sc, cax = ax2, ticks = range(16), label = 'Class')
    name = "{preprocess}_{n_bands}".format(**hyperparams)
    plt.savefig(os.path.join(test_pre_fpath, name + "_scatter3d.png"))
    plt.show()

def draw_scatter_2d(img_vec, gt_vec, hyperparams = {}):
    gt_vec = np.where(gt_vec == 0, np.nan, gt_vec)
    cmap = plt.get_cmap('tab20', 16)
    # cmap = plt.get_cmap('viridis', 16)
    fig = plt.figure(figsize = (15, 15))
    gs = gridspec.GridSpec(2, 2, width_ratios = [10, 1], height_ratios = [10, 1])
    ax = plt.subplot(gs[0])
    sc = ax.scatter(img_vec[:, 0], img_vec[:, 1], c = gt_vec, cmap = cmap)
    ax2 = plt.subplot(gs[0, 1])
    plt.colorbar(sc, cax = ax2, ticks = range(16), label = 'Class')
    name = "{preprocess}_{n_bands}".format(**hyperparams)
    plt.savefig(os.path.join(test_pre_fpath, name + "_scatter2d.png"))
    plt.show()


if __name__ == '__main__':
    # measure_pre_time()
    hyperparams = {
            "dataset"   : "IndianPines",
            "preprocess": "ica",
            "n_bands"   : 3,
    }
    # test_pre(hyperparams)
    measure_pre_time(hyperparams = hyperparams)