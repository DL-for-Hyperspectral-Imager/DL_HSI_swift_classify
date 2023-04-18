from utils import *
from preprocess import *
from sklearn.metrics import classification_report, accuracy_score
# 全局变量信息

n_runs = 1
dataset_name = "IndianPines"
# 加载原始数据集
img, gt, label_values = load_dataset(dataset_name)

# 数据预处理
# 数据归一化
img = np.asarray(img, dtype=np.float32)
img = (img - np.min(img)) / (np.max(img) - np.min(img))

# PCA降维-波段数减少
img = pca_sklearn(img, 50)

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
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))

#


print("hello")








#
# # 比较测试结果与真值
# run_results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=N_CLASSES)
#
# prediction[mask] = 0
#
# display_predictions(color_prediction, viz, gt=convert_to_color(
#     test_gt), caption="Prediction vs. test ground truth")
#
# results.append(run_results)
# # show_results(run_results, viz, label_values=LABEL_VALUES)
# show_results_save(run_results, viz, label_values=LABEL_VALUES, output_filename=res_filename)