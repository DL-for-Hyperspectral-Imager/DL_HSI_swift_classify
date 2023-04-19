import sys
import torch
import sklearn
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.nn import init

#不同平台相对路径加载方式不同
if('win' in sys.platform):
    from datasets import Mydatasets
elif('linux' in  sys.platform):
    from .datasets import Mydatasets

#定义一个神经网络模型
class neural_network_model(nn.Module):
    """
    Neural mdoel by layort
    use a simple network to classify the img
    ouput shape (bsz,n_classes)
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False, p = 0.5):
        super(neural_network_model, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=p)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
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
    X_test  = kwargs['X_test']
    y_test  = kwargs['y_test']
    n_bands = kwargs['n_bands']
    n_classes = kwargs['n_classes']
    n_runs = kwargs['n_runs']
    if(name == "svm"):# 使用SVM进行分类
        #加载svm分类器
        svm_classifier = sklearn.svm.SVC(kernel='rbf', C=10, gamma=0.001)
        svm_classifier.fit(X_train, y_train)
        # 预测测试集
        y_pred = svm_classifier.predict(X_test)
        return y_pred
    elif(name == 'nn'):
        #初始化神经模型
        net = neural_network_model(n_bands,n_classes,dropout= True, p = 0.2).cuda()
        bsz = 50 #batch_size
        print("X_train.shape",X_train.shape)
        print("y_train.shape",y_train.shape)
        print("n_classes",n_classes)
        #加载数据集,这里定义了张量tensor
        datasets = Mydatasets(X_train,y_train,bsz)
        #放入dataloader
        batch_loader = DataLoader(datasets,batch_size= bsz)
        #定义优化器
        optimizer = optim.AdamW(net.parameters(),lr = 0.01,weight_decay= 0.1)
        
        criterion =  nn.CrossEntropyLoss()
        
        for _ in tqdm(range(n_runs)):
            loss_avg = 0
            nums = 0
            for batch_X,batch_y in batch_loader:
                #看看训练集是否有问题
                if(any(batch_y[batch_y>n_classes])):
                    print("出现了大于%d的标签,错误！！！"%n_classes)
                    #print("batch_y[batch_y>n_classes]",batch_y[batch_y>n_classes][0])
                    continue
                #放入网络里面
                pred_classes = net(batch_X.cuda())
                nums += 1
                loss = criterion(pred_classes,batch_y.cuda())
                loss_avg += loss.item()
                #反向梯度传播
                loss.backward()
                #梯度优化
                optimizer.step()
            print("\nloss:%.2f"%(loss_avg/nums))
        #训练完后放入test进行测试
        
        y_pred  = net(torch.Tensor(X_test).cuda())
        y_pred = torch.topk(y_pred,k=1).indices
        return y_pred.cpu().numpy()