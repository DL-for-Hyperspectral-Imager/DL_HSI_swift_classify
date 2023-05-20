# -*- coding: utf-8 -*-

import sys
import torch
import sklearn
import numpy as np
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as nnFunc
from tqdm import trange
from torch.utils.data.dataloader import DataLoader
from torch.nn import init

from datasets import Mydatasets
# added by mangp, to solve a bug of sklearn
from sklearn import neighbors
# added by mangp, to save the classifier
import joblib
import os
import math

pwd = os.getcwd()
sklearn_clf = {"svm", "nearest"}


def train(hyperparams, **kwargs):
    """
    用于训练的函数
    :param :name the name of model
    :param :kwargs other args in form of dict
    :return y_pred the predict result of X_test
    """
    model_name = hyperparams["model"].lower()
    n_bands = hyperparams["n_bands"]
    n_classes = hyperparams["n_classes"]
    n_runs = hyperparams["n_runs"]

    X_train = kwargs['X_train']
    y_train = kwargs['y_train']

    load_model = hyperparams["load_model"]

    # 将训练好的分类器保存在classifier文件夹下面
    clf_folder = 'classifier'

    sklearn_clf_save_folder = os.path.join(pwd, "..", clf_folder)
    torch_model_save_folder = os.path.join(pwd, "..", clf_folder, 'torch_model', model_name)

    if load_model:
        if model_name in sklearn_clf:
            path = os.path.join(sklearn_clf_save_folder, model_name + '_' + 'clf')
            clf = joblib.load(path)
            return clf
        else:
            model = get_model(model_name, **hyperparams)
            model.load_state_dict(
                    torch.load(os.path.join(torch_model_save_folder, model_name + '_' + 'runs' + str(n_runs) + '.pth')))
            return model

    if model_name == "svm":  # 使用SVM进行分类
        clf = train_svm(X_train, y_train)
        joblib.dump(clf, os.path.join(sklearn_clf_save_folder, model_name + '_' + 'clf'))
        return clf
    elif model_name == 'nearest':
        clf = train_knn(X_train, y_train)
        joblib.dump(clf, os.path.join(sklearn_clf_save_folder, model_name + '_' + 'clf'))
        return clf
    else:  # pytorch model
        bsz = 1000  # batch_size
        model = get_model(model_name, **hyperparams)

        datasets = Mydatasets(X_train, y_train, bsz)  # 加载数据集,这里定义了张量tensor
        batch_loader = DataLoader(datasets, batch_size = bsz, shuffle = True)  # 放入dataloader

        optimizer = optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.01)  # 定义优化器

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')  # 定义学习率衰减策

        criterion = nn.CrossEntropyLoss()  # 定义损失函数

        t = trange(n_runs, desc = 'Runs')
        for run in t:
            loss_avg = 0  # 记录每个epoch的平均损失
            nums = 0  # 记录每个epoch的样本数
            optimizer.zero_grad()  # 必须要清零梯度
            for batch_X, batch_y in batch_loader:
                if any(batch_y[batch_y > n_classes]):  # 检查训练集是否有问题
                    print(f"出现了大于{n_classes}的标签,错误！！！")
                    continue
                # 输入网络进行训练
                pred_classes = model(batch_X.cuda())
                loss = criterion(pred_classes, batch_y.cuda().long())
                loss_avg += loss.item()
                nums += 1
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重
            scheduler.step(loss_avg / nums)  # 更新学习率
            t.set_postfix(loss = loss_avg / nums, learning_rate = optimizer.param_groups[0]['lr'])
        if isinstance(model, nn.Module):
            os.makedirs(torch_model_save_folder, exist_ok = True)
            torch.save(
                    model.state_dict(),
                    os.path.join(torch_model_save_folder, model_name + '_' + 'runs' + str(n_runs) + '.pth'))
        return model


def get_model(model_name, **kwargs):
    n_bands = kwargs["n_bands"]
    n_classes = kwargs["n_classes"]
    if model_name == 'nn':
        model = FNN(n_bands, n_classes, dropout = True, p = 0.5).cuda()
    elif model_name == 'cnn':
        model = CNN(n_bands, n_classes).cuda()
    else:
        raise KeyError("{} model is unknown.".format(model_name))
    return model


# 定义一个全连接神经网络模型
# channels - 2048 - 4096 - 2048 - n_classes
class FNN(nn.Module):
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

    def __init__(self, n_channels, n_classes, dropout = False, p = 0.2):
        super(FNN, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p = p)

        self.fc1 = nn.Linear(n_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = x.to(torch.float32)  ##
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


class CNN(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1,1, self.n_channels)
            x = self.conv(x)
            x = self.pool(x)
        return x.numel()
    def __init__(self, n_channels, n_classes, kernel_size = None, pool_size = None):
        super(CNN, self).__init__()
        if kernel_size is None:
            kernel_size = math.ceil(n_channels / 9)  # 200/9 = 23
        if pool_size is None:
            pool_size = math.ceil(kernel_size / 5)  # 23/5 = 5
        self.n_channels = n_channels
        self.conv = nn.Conv1d(
                            in_channels = 1,
                            out_channels = 20,
                            kernel_size = kernel_size)
        self.pool = nn.MaxPool1d(kernel_size = pool_size)
        self.features_size = self._get_final_flattened_size()
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x




def train_svm(X_train, y_train):
    # 加载svm分类器
    # svm_classifier = sklearn.svm.SVC(kernel='rbf', C=10, gamma=0.001)
    svm_classifier = sklearn.svm.SVC(class_weight = 'balanced', kernel = 'rbf', C = 10, gamma = 0.001)
    svm_classifier.fit(X_train, y_train)
    return svm_classifier


def train_knn(X_train, y_train):
    # 加载knn分类器
    knn_classifier = sklearn.neighbors.KNeighborsClassifier()
    # 通过选择最佳的邻居数量来执行knn分类器的参数调整
    # verbose为0表示不输出训练进度和信息
    knn_classifier = sklearn.model_selection.GridSearchCV(
            knn_classifier, {"n_neighbors": [1, 3, 5, 10, 20]}, verbose = 0, n_jobs = 4)
    knn_classifier.fit(X_train, y_train)
    return knn_classifier


def predict(model_name, clf, X_test):
    """
    用于预测的函数
    :param :model_name the name of model
    :param :clf the model
    :param :X_test the test data
    :return y_pred the predict result of X_test
    """
    model_name = model_name.lower()
    if model_name == "svm":  # 使用SVM进行分类
        y_pred = clf.predict(X_test)
    elif model_name == 'nearest':
        y_pred = clf.predict(X_test)
    elif model_name in ['nn', 'cnn']:
        y_pred = clf(torch.Tensor(X_test).cuda())
        y_pred = torch.topk(y_pred, k = 1).indices
        y_pred = y_pred.cpu().numpy()
    else:
        print("The model name is wrong")
        y_pred = None
    return y_pred.reshape(-1)
