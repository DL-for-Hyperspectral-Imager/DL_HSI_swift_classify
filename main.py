# -*- coding: utf-8 -*-

"""
变量命名规范
*img, *gt 等都是(145,145,200)的numpy数组
*X 是(145*145,200)的numpy数组
*y 是(145*145,)的numpy数组
"""

from utils import *
from model import *
from preprocess import *
import argparse

import time


# 这里定义一个main函数作为整个项目的入口，只有作为脚本时这个文件才会运行

def main():
    # 获取args
    args = args_init()
    hyperparams = args.__dict__

    # 加载原始数据集
    hsi_img, gt, label_values, ignored_labels = load_dataset(hyperparams["dataset"])
    hyperparams["n_classes"] = len(label_values)
# 数据预处理
    img = preprocess(hsi_img, gt, preprocess_name=hyperparams["preprocess"], n_bands=hyperparams["n_bands"])

    hyperparams["height"], hyperparams["width"], hyperparams["n_bands"] = img.shape

    # 划分训练集和测试集
    train_gt, test_gt = split_train_test(gt, hyperparams["training_sample"])
    # 依据train_gt构建训练集
    X_train, y_train = build_dataset(img, train_gt)
    # 训练
    start_train = time.time()
    clf = train(hyperparams, X_train=X_train, y_train=y_train)
    end_train = time.time()
    # 预测
    start_pred = time.time()
    y_img_pred = predict(hyperparams["model"], clf, img.reshape(-1, hyperparams["n_bands"]))
    end_pred = time.time()

    run_results = metrics(y_img_pred, gt.reshape(-1), ignored_labels, hyperparams["n_classes"])

    # 可视化与结果输出
    visualize(hsi_img, gt, y_img_pred.reshape(hyperparams["height"], hyperparams["width"]))
    show_results(args, run_results, label_values)

    print('Training time: %.5fs' % (end_train - start_train))
    print('Predicting time: %.5fs' % (end_pred - start_pred))

def args_init():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_runs', type=int, default=1, help='number of runs')
    parser.add_argument('--dataset', type=str, default='IndianPines', help='dataset name')
    parser.add_argument('--preprocess', type=str, default='', help='preprocess name')
    parser.add_argument('--model', type=str, default='SVM', help='model name')
    parser.add_argument('--training_sample', type=float, default=0.1, help='training sample')
    parser.add_argument('--n_bands', type=int, default=50, help='number of bands')
    return parser.parse_args()

if __name__ == "__main__":
    main()
