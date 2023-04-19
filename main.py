from utils import *
from preprocess import *
from sklearn.metrics import classification_report, accuracy_score
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--n_runs', type=int, default=1, help='number of runs')
parser.add_argument('--dataset', type=str, default='IndianPines', help='dataset name')
parser.add_argument('--preprocess', type=str, default='LDA', help='preprocess name')
parser.add_argument('--model', type=str, default='SVM', help='model name')
args = parser.parse_args()
# 获得全局变量信息
n_runs = args.n_runs
dataset_name = args.dataset
preprocess_name = args.preprocess
model_name = args.model
# 加载原始数据集
img, gt, label_values = load_dataset(dataset_name)

# 数据预处理
# 数据归一化
img = np.asarray(img, dtype=np.float32)
img = (img - np.min(img)) / (np.max(img) - np.min(img))
# 根据预处理方法进行预处理

    
print("-----------------------------")

img = preprocess(img, preprocess_name, gt) #make little changes

n_classes = len(label_values)
n_bands = img.shape[-1]

# 划分训练集和测试集
# XX_gt: 145*145，存有坐标处像素的真值，为0代表未选择该像素
train_gt, test_gt = split_train_test(gt, 0.7)
# 依据train_gt构建训练集
X_train, y_train = build_set(img, train_gt)
X_test, y_test = build_set(img, test_gt)
# 使用SVM进行分类
svm_classifier = sklearn.svm.SVC(kernel='rbf', C=10, gamma=0.001)
svm_classifier.fit(X_train, y_train)
# 预测测试集
y_pred = svm_classifier.predict(X_test)

# 输出分类报告和准确率
report = classification_report(y_test, y_pred, zero_division=1)
accuracy = accuracy_score(y_test, y_pred)
print("Classification Report:\n", report)
print("Accuracy: ", accuracy)
    
print("------------------------------")






