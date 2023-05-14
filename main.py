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
# 注意此处修改，main函数添加了一个参数，为一字典，若没有从命令行输入参数，
# 则直接使用args中指定的值，同时方便使用其他python脚本进行测试。
# 改动：将run_results, Training_time, Predicting_time作为返回值返回，方便使用脚本多次运行和保存结果
# 改动：添加了一个输出开关选项，可以选择关闭输出
def main(show_results_switch, **kwargs):
    """
    Args:
        show_results_switch: if True, then print result, otherwise no print(usually for testing)
        kwargs: if args were not given in command line, then use kwargs(usually for testing)
    Returns:
        run_results(for analyzing)
        Training_time
        Predicting_time
    """
    # 获取args
    args = args_init(**kwargs)
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
    visualize(hsi_img, gt, y_img_pred.reshape(hyperparams["height"], hyperparams["width"]),hyperparams["img_path"],hyperparams["model"]+'_'+hyperparams["preprocess"]+'_'+str(hyperparams["n_bands"])
              +'_')

    Training_time = end_train - start_train
    Predicting_time = end_pred - start_pred

    #下方修改，若输出开关关闭则关闭输出
    if(show_results_switch == True):
        show_results(args, run_results, label_values)
        print('Training time: %.5fs' % (Training_time))
        print('Predicting time: %.5fs' % (Predicting_time))

    
    #下方改动,将重要结果返回
    return run_results, Training_time, Predicting_time

def args_init(**kwargs):
    # 添加需要解析的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_runs', type=int, default=1, help='number of runs')
    parser.add_argument('--dataset', type=str, default='IndianPines', help='dataset name')
    parser.add_argument('--preprocess', type=str, default='0', help='preprocess name')
    parser.add_argument('--model', type=str, default='svm', help='model name')
    parser.add_argument('--training_sample', type=float, default=0.1, help='training sample')
    parser.add_argument('--n_bands', type=int, default=50, help='number of bands')
    parser.add_argument('--img_path',type=str,default='result',help='path for saved img')
    #此处修改，若没有从命令行输入参数，则手动设置参数
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"存在未知参数：{unknown_args}")
    else:
        print("没有从命令行输入的参数")
        #此时使用kwargs中传递的参数，方便快速多次测试
        args = argparse.Namespace()
        args.n_runs          = kwargs['n_runs']
        args.dataset         = kwargs['dataset']
        args.preprocess      = kwargs['preprocess']
        args.model           = kwargs['model']
        args.training_sample = kwargs['training_sample']
        args.n_bands         = kwargs['n_bands']
        args.img_path        = kwargs['img_path']
    return args
if __name__ == "__main__":
    main()
