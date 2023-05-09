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
from sklearn.metrics import classification_report, accuracy_score
import argparse
import PIL.Image as Image
import time


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


# 这里定义一个main函数作为整个项目的入口，只有作为脚本时这个文件才会运行

def main():
    # 获取args
    args = args_init()
    # 获得全局变量信息
    n_runs = args.n_runs
    dataset_name = args.dataset
    preprocess_name = args.preprocess
    model_name = args.model
    sample_rate = args.training_sample
    # 加载原始数据集
    hsi_img, gt, label_values = load_dataset(dataset_name)

# 数据预处理
    # 数据归一化
    img = np.asarray(hsi_img, dtype=np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    # 根据预处理方法进行预处理
    img = preprocess(img, gt, preprocess_name, n_bands=args.n_bands)

    n_classes = len(label_values)
    height, width, n_bands = img.shape

# 划分训练集和测试集
    # XX_gt: 145*145，存有坐标处像素的真值，为0代表未选择该像素
    train_gt, test_gt = split_train_test(gt, sample_rate)
    # 依据train_gt构建训练集
    X_train, y_train = build_set(img, train_gt)
    X_test, y_test = build_set(img, test_gt)
    # 训练模型
    start_train = time.time()
    clf = train(model_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                n_classes=n_classes, n_bands=n_bands, n_runs=n_runs)
    end_train = time.time()
    # 预测
    start_pred = time.time()
    y_img_pred = predict(model_name, clf, img.reshape((height * width, n_bands)))
    end_pred = time.time()

    y_img_pred = y_img_pred.reshape(height * width)
    run_results = metrics(y_img_pred, gt.reshape(height * width), [0], n_classes)

    # 输出分类报告和准确率
    visualize(hsi_img, gt, y_img_pred.reshape((height, width)))
    show_results(args, run_results, label_values)

    print('Training time: %.5fs' % (end_train - start_train))
    print('Predicting time: %.5fs' % (end_pred - start_pred))

if __name__ == "__main__":
    main()
