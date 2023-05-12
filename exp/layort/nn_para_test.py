# -*- coding: utf-8 -*-

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
    best_args = (-1,-1,-1)
    best_accuracy = 0
    for n_brands_ in []:
        for n_runs in []:
            for training_sample in []:
                args["n_brands"] = n_brands_
                args["n_runs"] = n_runs
                args["training_sample"] = training_sample
                # 训练
                clf = train(hyperparams, X_train=X_train, y_train=y_train)
                # 预测
                y_img_pred = predict(hyperparams["model"], clf, img.reshape(-1, hyperparams["n_bands"]))

                run_results = metrics(y_img_pred, gt.reshape(-1), ignored_labels, hyperparams["n_classes"])
                if(run_results["Accuracy"] > best_accuracy):
                    best_args = (n_brands_,n_runs,training_sample)
    print("best_args:",best_args)

    

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