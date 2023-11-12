import torch
import torch.nn as nn
import torch.nn.functional as nnFunc


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
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def __init__(self, n_channels, n_classes, dropout=False, p=0.2):
        super(FNN, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=p)

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
