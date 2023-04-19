from utils import *
from model import train
from preprocess import *
from sklearn.metrics import classification_report, accuracy_score
import argparse


def args_init():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_runs', type=int, default=1, help='number of runs')
    parser.add_argument('--dataset', type=str, default='IndianPines', help='dataset name')
    parser.add_argument('--preprocess', type=str, default=None, help='preprocess name')
    parser.add_argument('--model', type=str, default='SVM', help='model name')
    parser.add_argument('--sample_rate', type=float, default=0.3, help='sample rate')
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
    sample_rate = args.sample_rate
    # 加载原始数据集
    img, gt, label_values = load_dataset(dataset_name)

    # 数据预处理
    # 数据归一化
    img = np.asarray(img, dtype=np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    # 根据预处理方法进行预处理
    img = preprocess(img, preprocess_name)

    n_classes = len(label_values)
    n_bands = img.shape[-1]

    # 划分训练集和测试集
    # XX_gt: 145*145，存有坐标处像素的真值，为0代表未选择该像素
    train_gt, test_gt = split_train_test(gt, sample_rate)
    # 依据train_gt构建训练集
    X_train, y_train = build_set(img, train_gt)
    X_test, y_test = build_set(img, test_gt)

    y_pred = train(model_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                   n_classes=n_classes, n_bands=n_bands, n_runs=n_runs)

    # 输出分类报告和准确率
    report = classification_report(y_test, y_pred, zero_division=1)
    accuracy = accuracy_score(y_test, y_pred)
    print("Classification Report:\n", report)
    print("Accuracy: ", accuracy)


if __name__ == "__main__":
    main()
