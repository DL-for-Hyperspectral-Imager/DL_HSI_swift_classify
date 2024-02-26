# -*- coding: utf-8 -*-

"""
变量命名规范
*img, *gt 等都是(145,145,200)的numpy数组
*X 是(145*145,200)的numpy数组
*y 是(145*145,)的numpy数组
"""

from utils import (
    metrics_beta,
    get_img_pred,
    save_pred,
    show_result,
)
from train import train, predict
from preprocess import preprocess
from dataset import load_dataset, split_train_test, build_Xy

import argparse
import numpy as np
import torch
import os
import time


def try_gpu():
    if torch.cuda.device_count() >= 1:
        return torch.device(f"cuda:{0}")
    return torch.device("cpu")


# setting random seed, for replicate
def seed_torch(seed=1029):
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 这里定义一个main函数作为整个项目的入口，只有作为脚本时这个文件才会运行
# 注意此处修改，main函数添加了一个参数，为一字典，若没有从命令行输入参数，
# 则直接使用args中指定的值，同时方便使用其他python脚本进行测试。
# 改动：将run_results, Training_time, Predicting_time作为返回值返回，方便使用脚本多次运行和保存结果
# 改动：添加了一个输出开关选项，可以选择关闭输出
def main(is_show=True, hparams={}):
    """
    Args:
        show_results_switch: if True, then print result, otherwise no print(usually for testing)
        kwargs: if args were not given in command line, then use kwargs(usually for testing)
    Returns:
        run_results(for analyzing)
        training_time
        predicting_time
    """

    # >>> 加载原始数据集
    hsi_img, gt, hparams["labels"], hparams["ignored_labels"] = load_dataset(
        hparams["dataset"]
    )
    hparams["n_classes"] = len(hparams["labels"])

    # >>> 数据预处理
    START_preprocess = time.time()
    img = preprocess(
        hsi_img=hsi_img,
        gt=gt,
        name=hparams["preprocess"],
        n_bands=hparams["t_bands"],
    )
    END_preprocess = time.time()

    hparams["shape"] = img.shape # not used

    # >>> Train

    ## 划分训练集和测试集
    train_gt, test_gt = split_train_test(gt, hparams["train_rate"])

    ## 依据train_gt构建训练集
    if hparams["model"] in ["CNN2D"]:
        X_train, y_train, _ = build_Xy(img, train_gt, True, hparams["patch_size"])
        X_test, y_test, _ = build_Xy(img, test_gt, True, hparams["patch_size"])
    else:
        X_train, y_train, _ = build_Xy(img, train_gt)
        X_test, y_test, _ = build_Xy(img, test_gt)


    print(
        "Model: {model}, n_runs: {n_runs}, batch_size: {batch_size}, patch_size: {patch_size}".format(
            **hparams
        )
    )

    START_train = time.time()
    clf, record = train(
        hparams,
        X_train,
        y_train,
        X_test,
        y_test,
        device=try_gpu(),
    )
    END_train = time.time()

    # >>> 预测
    START_pred = time.time()

    if hparams["model"] in ["CNN2D"]:
        X_test_all, y_test_all, labels_indices = build_Xy(
            img, gt, True, hparams["patch_size"]
        )
    else:
        X_test_all, y_test_all, labels_indices = build_Xy(img, gt)

    y_test_all_pred = predict(
        model_name=hparams["model"], clf=clf, X_test=X_test_all, device=try_gpu()
    )
    END_pred = time.time()

    # >>> evaluate
    pred_img = get_img_pred(y_test_all_pred, labels_indices, gt.shape)

    result = metrics_beta(y_test_all, y_test_all_pred, hparams["n_classes"])

    # >>> save res
    name = "pred-{dataset}_{train_rate}-{preprocess}_{t_bands}-".format(**hparams)
    name += "{model}_runs{n_runs}_bsz{batch_size}_psz{patch_size}".format(**hparams)
    name += f"_acy{result['accuracy']}.png"

    res_path = os.path.join(
        os.getcwd(),
        "..",
        hparams["res_folder"],
        "{preprocess}_{model}".format(**hparams),
    )
    save_pred(
        pred_img=pred_img,
        res_path=res_path,
        n_classes=hparams["n_classes"],
        name=name,
    )

    result["preprocess_time"] = END_preprocess - START_preprocess
    result["train_time"] = END_train - START_train
    result["predict_time"] = END_pred - START_pred

    if is_show:
        show_result(result, hparams)

    return result


def args_init(**kwargs):
    # 添加需要解析的参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", "-data", type=str, default="IndianPines", help="dataset name"
    )
    parser.add_argument(
        "--train_rate", "-rate", type=float, default=0.3, help="training rate"
    )
    # preprocess and t_bands
    parser.add_argument(
        "--preprocess", "-pre", type=str, default="nopre", help="preprocess name"
    )
    parser.add_argument(
        "--t_bands",
        "-nb",
        type=int,
        default=0,  # 0 表示不降维
        help="number of bands",
    )
    # model and n_runs
    parser.add_argument("--model", "-m", type=str, default="svm", help="model name")
    parser.add_argument("--n_runs", "-nr", type=int, default=1, help="number of runs")
    # res_folder and load_model
    parser.add_argument(
        "--res_folder",
        "-folder",
        type=str,
        default="result",  # 相对于主目录的文件夹相对路径
        help="folder for saved img",
    )
    parser.add_argument(
        "--load_model", "-lm", type=bool, default=False, help="if load model"
    )
    parser.add_argument(
        "--patch_size", "-psz", type=int, default=10, help="patch size of slide windows"
    )
    parser.add_argument(
        "--batch_size", "-bsz", type=int, default=1000, help="batch size"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 获取args
    # args = args_init()
    # hparams = vars(args)
    seed_torch(42)

    hparams = {
        "dataset": "IndianPines",
        "train_rate": 0.9,
        "preprocess": "PCA",
        "t_bands": 200,
        "model": "NN",
        "n_runs": 12500,
        "res_folder": "result",
        "batch_size": 1000,
        "patch_size": 10,
        "load_model": False,
    }
    main(is_show=True, hparams=hparams)
