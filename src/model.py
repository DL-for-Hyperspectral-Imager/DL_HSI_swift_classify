# -*- coding: utf-8 -*-

import sys
import torch
import sklearn
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.functional as nnFunc
from tqdm import trange
from torch.utils.data.dataloader import DataLoader
from torch.nn import init

from datasets import Mydatasets,CNNDatasets
# added by mangp, to solve a bug of sklearn
from sklearn import neighbors
# added by mangp, to save the classifier
import joblib
import os
import math

pwd = os.getcwd()
sklearn_clf = {"svm", "knn"}


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
    elif model_name == 'knn':
        clf = train_knn(X_train, y_train)
        joblib.dump(clf, os.path.join(sklearn_clf_save_folder, model_name + '_' + 'clf'))
        return clf
    elif model_name in ['nn','cnn1d']:  # pytorch model
        batch_size = hyperparams['batch_size']  # batch_size
        model = get_model(model_name, hyperparams)

        datasets = Mydatasets(X_train, y_train)  # 加载数据集,这里定义了张量tensor
        batch_loader = DataLoader(datasets, batch_size = batch_size, shuffle = True)  # 放入dataloader

        optimizer = optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.01)  # 定义优化器

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')  # 定义学习率衰减策

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
            # scheduler.step(loss_avg / nums)  # 更新学习率
            t.set_postfix(loss = loss_avg / nums, learning_rate = optimizer.param_groups[0]['lr'])
        if isinstance(model, nn.Module):
            os.makedirs(torch_model_save_folder, exist_ok = True)
            torch.save(
                    model.state_dict(),
                    os.path.join(torch_model_save_folder, model_name + '_' + 'runs' + str(n_runs) + '.pth'))
        return model
    elif model_name == 'cnn2d':
        batch_size = hyperparams['batch_size']  # batch_size
        model = get_model(model_name, hyperparams)
        
        datasets = Mydatasets(X_train, y_train)  # 加载数据集,这里定义了张量tensor

        batch_loader = DataLoader(datasets, batch_size = batch_size, shuffle = True)  # 放入dataloader

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


def get_model(model_name, hyperparams):
    n_bands = hyperparams["n_bands"]
    n_classes = hyperparams["n_classes"]
    if model_name == 'nn':
        model = FNN(n_bands, n_classes, dropout = True, p = 0.5).cuda()
    elif model_name == 'cnn1d':
        # if(hyperparams['n_bands'])
        model = CNN1D(n_bands, n_classes).cuda()
    elif model_name == 'cnn2d':
        model = CNN2D(n_bands, n_classes, patch_size=hyperparams['patch_size']).cuda()
    else:
        raise KeyError("{} model is unknown.".format(model_name))
    return model


# 定义一个全连接神经网络模型
# channels - 2048 - 4096 - 2048 - n_classes
class FNN(nn.Module):
    """
    Neural mdoel by layort
    use a simple network to classify the img
    ouput shape (batch_size,n_classes)
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

# CNN1D对波段进行卷积，输入channels数量>1，建议输入channels>50
class CNN1D(nn.Module):
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
        super(CNN1D, self).__init__()
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



class CNN2D(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=9):
        super(CNN2D, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.aux_loss_weight = 1

        # "W1 is a 3x3xB1 kernel [...] B1 is the number of the output bands for the convolutional
        # "and pooling layer" -> actually 3x3 2D convolutions with B1 outputs
        # "the value of B1 is set to be 80"
        self.conv1 = nn.Conv2d(input_channels, 80, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))

        self.features_sizes = self._get_sizes()

        self.fc_enc = nn.Linear(self.features_sizes[2], n_classes)

        self.apply(self.weight_init)

    def _get_sizes(self):
        x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size))
        x = F.relu(self.conv1(x))
        _, c, w, h = x.size()
        size0 = c * w * h

        x = self.pool1(x)
        _, c, w, h = x.size()
        size1 = c * w * h

        _, c, w, h = x.size()
        size2 = c * w * h

        return size0, size1, size2

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x_conv1 = self.conv1(x)
        x = x_conv1
        x_pool1 = self.pool1(x)
        x = x_pool1
        x_enc = F.relu(x).contiguous().view(-1, self.features_sizes[2])
        x = x_enc
        x_classif = self.fc_enc(x)
        return x_classif

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
        knn_classifier, {"n_neighbors": [1, 3, 5, 7, 9, 10, 13, 15, 20], \
                        "metric":['euclidean', 'manhattan', 'chebyshev']},\
                        verbose = 0, n_jobs = 8)
    knn_classifier.fit(X_train, y_train)
    print("The best k is %d:" %knn_classifier.best_params_['n_neighbors'])
    return knn_classifier


def predict(model_name, clf, X_test, patch_size, vector_mask = None):
    """
    用于预测的函数
    :param :model_name the name of model
    :param :clf the model
    :param :X_test the test data, shape = (n_samples, n_features)
    :return y_pred the predict result of X_test
    """
    model_name = model_name.lower()
    if model_name == "svm":  # 使用SVM进行分类
        y_pred = clf.predict(X_test)
    elif model_name == 'knn':
        y_pred = clf.predict(X_test)
    elif model_name in ['nn', 'cnn1d']:
        y_pred = clf(torch.Tensor(X_test).cuda())
        y_pred = torch.topk(y_pred, k = 1).indices
        y_pred = y_pred.cpu().numpy()
    elif model_name == 'cnn2d':
        slide_windows_datasets = CNNDatasets(X_test, np.ones((len(X_test))), patch_size= patch_size)
        slide_windows_dataloader = DataLoader(slide_windows_datasets,batch_size=len(slide_windows_datasets))
        with torch.no_grad():
            for X_test_window, y_test in slide_windows_dataloader:
                y_pred = clf(torch.Tensor(X_test_window).cuda())
                y_pred = torch.topk(y_pred, k=1).indices
                y_pred = y_pred.cpu().numpy()
    else:
        print("The model name is wrong")
        y_pred = None
    y_pred[vector_mask==1] = 0
    return y_pred

