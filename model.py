# -*- coding: utf-8 -*-

import sys
import torch
import sklearn
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.nn.functional as nnFunc
from datasets import Mydatasets
from tqdm import trange
from torch.utils.data.dataloader import DataLoader
from torch.nn import init

# 不同平台相对路径加载方式不同
if ('win' in sys.platform):
    from datasets import Mydatasets
elif ('linux' in sys.platform):
    from datasets import Mydatasets
# added by mangp, to solve a bug of sklearn
from sklearn import neighbors


# 定义一个神经网络模型


# 定义一个全连接神经网络模型
# channels - 2048 - 4096 - 2048 - n_classes
class neural_network_model(nn.Module):
    """
    Neural mdoel by layort
    use a simple network to classify the img
    ouput shape (bsz,n_classes)
    use a simple full connected network to classify the img
    ouput shape (batch_size,n_classes)
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, n_channels, n_classes, dropout=False, p=0.2):
        super(neural_network_model, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=p)

        self.fc1 = nn.Linear(n_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = nnFunc.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = nnFunc.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = nnFunc.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc4(x)
        return x


def train(name, **kwargs):
    """
    用于训练的函数
    :param :name the name of model
    :param :kwargs other args in form of dict
    :return y_pred the predict result of X_test
    """
    X_train = kwargs['X_train']
    y_train = kwargs['y_train']
    X_test = kwargs['X_test']
    y_test = kwargs['y_test']
    n_bands = kwargs['n_bands']
    n_classes = kwargs['n_classes']
    n_runs = kwargs['n_runs']
    name = name.lower()
    if name == "svm":  # 使用SVM进行分类
        clf = train_svm(X_train, y_train)
        return clf
    elif name == 'nearest':
        clf = train_knn(X_train, y_train)
        return clf
    elif name == 'nn':
        # 初始化神经模型
        net = neural_network_model(n_bands, n_classes, dropout=True, p=0.5).cuda()
        bsz = 1000  # batch_size
        print("X_train.shape", X_train.shape)
        print("y_train.shape", y_train.shape)
        print("n_classes", n_classes)
        # 加载数据集,这里定义了张量tensor
        datasets = Mydatasets(X_train, y_train, bsz)
        # 放入dataloader
        batch_loader = DataLoader(datasets, batch_size=bsz, shuffle=True)
        # 定义优化器
        optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)

        criterion = nn.CrossEntropyLoss()

        t = trange(n_runs, desc='Runs')
        for run in t:
            loss_avg = 0
            nums = 0
            for batch_X, batch_y in batch_loader:
                # 检查训练集是否有问题
                if any(batch_y[batch_y > n_classes]):
                    print(f"出现了大于{n_classes}的标签,错误！！！")
                    continue
                # 输入网络进行训练
                pred_classes = net(batch_X.cuda())
                nums += 1
                loss = criterion(pred_classes, batch_y.cuda().long())
                loss_avg += loss.item()
                # 反向传播
                loss.backward()
                # 更新权重
                optimizer.step()
                # 更新进度条描述
                if nums % 25 == 0:
                    t.set_postfix(Loss=f'{loss_avg / nums:.2f}', refresh=True)
                    print()
        return net


def train_svm(X_train, y_train):
    # 加载svm分类器
    # svm_classifier = sklearn.svm.SVC(kernel='rbf', C=10, gamma=0.001)
    svm_classifier = sklearn.svm.SVC(class_weight='balanced', kernel='rbf', C=10, gamma=0.001)
    svm_classifier.fit(X_train, y_train)
    return svm_classifier


def train_knn(X_train, y_train):
    # 加载knn分类器
    knn_classifier = sklearn.neighbors.KNeighborsClassifier()
    knn_classifier = sklearn.model_selection.GridSearchCV(
        knn_classifier, {"n_neighbors": [1, 3, 5, 10, 20]}, verbose=5, n_jobs=4)
    knn_classifier.fit(X_train, y_train)
    return knn_classifier


def predict(name, clf, X_test):
    """
    用于预测的函数
    :param :name the name of model
    :param :clf the model
    :param :X_test the test data
    :return y_pred the predict result of X_test
    """
    name = name.lower()
    if (name == "svm"):  # 使用SVM进行分类
        y_pred = clf.predict(X_test)
    elif (name == 'nearest'):
        y_pred = clf.predict(X_test)
    elif (name == 'nn'):
        y_pred = clf(torch.Tensor(X_test).cuda())
        y_pred = torch.topk(y_pred, k=1).indices
        y_pred = y_pred.cpu().numpy()
    return y_pred
