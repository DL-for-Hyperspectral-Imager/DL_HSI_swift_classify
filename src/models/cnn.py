import torch
import torch.nn as nn
import torch.nn.functional as nnFunc
import math


# CNN1D对波段进行卷积，输入channels数量>1，建议输入channels>50
class CNN1D(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.uniform_(m.weight, -0.05, 0.05)
            nn.init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels)
            x = self.conv(x)
            x = self.pool(x)
        return x.numel()

    def __init__(self, n_channels, n_classes, kernel_size=None, pool_size=None):
        super(CNN1D, self).__init__()
        if kernel_size is None:
            kernel_size = math.ceil(n_channels / 10)
        if pool_size is None:
            pool_size = math.ceil(kernel_size / 5)
        self.n_channels = n_channels
        self.conv = nn.Conv1d(in_channels=1, out_channels=20, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.features_size = self._get_final_flattened_size()
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        x = x.float()
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
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=9, dtype=torch.float32):
        super(CNN2D, self).__init__()
        self.dtype = dtype
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.aux_loss_weight = 1

        self.conv1 = nn.Conv2d(input_channels, 80, (3, 3), dtype=dtype)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.features_sizes = self._get_sizes()
        self.fc_enc = nn.Linear(self.features_sizes[2], n_classes, dtype=dtype)

        self.apply(self.weight_init)

    def _get_sizes(self):
        x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size), dtype=self.dtype
        )
        x = nnFunc.relu(self.conv1(x))
        _, c, w, h = x.size()
        size0 = c * w * h
        x = self.pool1(x)
        _, c, w, h = x.size()
        size1 = c * w * h
        _, c, w, h = x.size()
        size2 = c * w * h
        return size0, size1, size2

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x_conv1 = self.conv1(x)
        x = x_conv1
        x_pool1 = self.pool1(x)
        x = x_pool1
        x_enc = nnFunc.relu(x).contiguous().view(-1, self.features_sizes[2])
        x = x_enc
        x_classif = self.fc_enc(x)
        return x_classif
