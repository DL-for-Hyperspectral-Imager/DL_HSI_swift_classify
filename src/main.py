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
import sys

# 这里定义一个main函数作为整个项目的入口，只有作为脚本时这个文件才会运行
# 注意此处修改，main函数添加了一个参数，为一字典，若没有从命令行输入参数，
# 则直接使用args中指定的值，同时方便使用其他python脚本进行测试。
# 改动：将run_results, Training_time, Predicting_time作为返回值返回，方便使用脚本多次运行和保存结果
# 改动：添加了一个输出开关选项，可以选择关闭输出
def main(show_results_switch = True, hyperparams = {}):
    """
    Args:
        show_results_switch: if True, then print result, otherwise no print(usually for testing)
        kwargs: if args were not given in command line, then use kwargs(usually for testing)
    Returns:
        run_results(for analyzing)
        training_time
        predicting_time
    """
    # 加载原始数据集
    hsi_img, gt, label_values, ignored_labels, rgb_bands = load_dataset(hyperparams["dataset"])
    rgb_IMG = hsi2rgb(hsi_img, rgb_bands = rgb_bands)
    path = os.path.join(os.getcwd(), "..", "Datasets", "IndianPines", "rgb.png")
    rgb_IMG.save(path)

    hyperparams["n_classes"] = len(label_values)
    start_preprocess = time.time()
    img = preprocess(
            hsi_img = hsi_img,
            gt = gt,
            preprocess_name = hyperparams["preprocess"],
            n_bands = hyperparams["n_bands"],
    )  # 数据预处理
    end_preprocess = time.time()
    hyperparams["height"], hyperparams["width"], hyperparams["n_bands"] = img.shape

    train_gt, test_gt = split_train_test(gt, hyperparams["training_rate"])  # 划分训练集和测试集

    if(hyperparams['model'] == 'cnn2d'):
        X_train, y_train = build_dataset_cnn2d(img, train_gt,hyperparams["patch_size"] )  # 依据train_gt构建训练集
    else:
        X_train, y_train = build_dataset(img, train_gt)  # 依据train_gt构建训练集

    start_train = time.time()
    clf = train(hyperparams, X_train = X_train, y_train = y_train)  # 训练
    end_train = time.time()
    # 预测
    start_pred = time.time()
    y_img_pred = predict(
            model_name = hyperparams["model"],
            clf = clf,
            X_test = img.reshape(-1, hyperparams["n_bands"]),
            patch_size = hyperparams["patch_size"]
    )
    end_pred = time.time()
    run_results = metrics(
            prediction = y_img_pred,
            target = gt.reshape(-1),
            ignored_labels = ignored_labels,
            n_classes = hyperparams["n_classes"]
    )

    # 可视化与结果输出
    visualize(
            hsi_img = hsi_img,
            gt = gt,
            pred_img = y_img_pred.reshape(hyperparams["height"], hyperparams["width"]),
            n_classes = hyperparams["n_classes"],
            img_path = hyperparams["img_path"] + r'/' + hyperparams['model'] + '_' + hyperparams['preprocess'],
            name = hyperparams["model"] + "_" + hyperparams["preprocess"] + "_" + str(hyperparams["n_bands"]) + "_",
    )

    preprocess_time = end_preprocess - start_preprocess
    training_time = end_train - start_train
    predicting_time = end_pred - start_pred
    
    run_results['preprocess_time'] = preprocess_time
    run_results['training_time'] = training_time
    run_results['predicting_time'] = predicting_time
    # 下方修改，若输出开关关闭则关闭输出
    if show_results_switch:
        show_results(args, run_results, label_values)
        print("Training time: %.5fs" % (training_time))
        print("Predicting time: %.5fs" % (predicting_time))

    # 下方改动,将重要结果返回一个字典
    return run_results


def args_init(**kwargs):
    # 添加需要解析的参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--dataset",
            type = str,
            default = "IndianPines",
            help = "dataset name")
    parser.add_argument(
            "--training_rate",
            type = float,
            default = 0.1,
            help = "training sample")
    # preprocess and n_bands
    parser.add_argument(
            "--preprocess",
            type = str,
            default = "0",
            help = "preprocess name")
    parser.add_argument(
            "--n_bands",
            type = int,
            default = 50,
            help = "number of bands")
    # model and n_runs
    parser.add_argument(
            "--model",
            type = str,
            default = "svm",
            help = "model name")
    parser.add_argument(
            "--n_runs",
            type = int,
            default = 1,
            help = "number of runs")
    # img_path and load_model
    parser.add_argument(
            "--img_path",
            type = str,
            default = "result",
            help = "path for saved img")
    parser.add_argument(
            "--load_model",
            type = str,
            default = None,
            help = "if load model")
    parser.add_argument(
            "--patch_size",
            type = int,
            default = 10,
            help = "patch size of slide windows")
    parser.add_argument(
            "--bsz",
            type = int,
            default = 1000,
            help = "batch size")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 获取args
    args = args_init()
    hyperparams = vars(args)
    main(show_results_switch = True,
         hyperparams = hyperparams)
