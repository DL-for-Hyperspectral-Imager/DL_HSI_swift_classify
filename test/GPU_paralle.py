# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:05:20 2023

@author: Cou'ra'ge
"""

import numpy as np
import torch

def lle(X, n_components, n_neighbors):
    """
    X:输入的样本，形式是N*D, N是样本数，D是样本特征维度
    n_components:表示降维到的维度数
    n_neighbors:表示LLE算法计算权重时的近邻数
    """
    X = X.reshape(-1, X.shape[-1])  # 将数据拉平为二维的输入矩阵，方便后续计算
    N, D = X.shape

    # 计算距离矩阵
    X = torch.tensor(X, dtype=torch.float32, device='cuda')  # 将输入样本转换为PyTorch Tensor类型，并将其放在GPU上
    X_square = torch.sum(X ** 2, dim=1)
    dists = X_square.reshape(-1, 1) + X_square.reshape(1, -1) - 2 * torch.matmul(X, X.t())
    dists = dists.cpu().numpy()

    # 2. 对于每个样本，找到其k个近邻，并计算权重矩阵。
    W = np.zeros((N, n_neighbors), dtype=np.float32)
    indices = np.zeros((N, n_neighbors), dtype=np.int32)
    for i in range(N):
        dists_i = dists[i]
        # 第i个样本距离其他样本的距离排序，并取前k大的邻居。
        indices_i = dists_i.argsort()[1:n_neighbors+1]  # 不包括自身
        indices[i] = indices_i
        # 计算最小化重构误差的权重。
        Xi = X[indices_i] - X[i]  # Xi是所有近邻与xi之间的差值
        G = torch.matmul(Xi, Xi.t())  # G = Xi * Xi.T, 是协方差矩阵
        # 设置误差的微小常数，提高算法的鲁棒性.
        G += 1e-3 * torch.eye(n_neighbors, device='cuda')
        w = torch.linalg.solve(G, torch.ones(n_neighbors, device='cuda'))
        w /= w.sum()
        W[i,:] = w.cpu().numpy()

    # 计算降维的权重矩阵并执行矩阵乘法。
    M = np.identity(N) - np.dot(W, W.T)
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(M.T, M))
    # 按升序对特征值排序，并选择最小的K-1个值对应的特征向量。
    indices = np.argsort(eigenvalues)[: n_components]
    components = eigenvectors[:, indices]
    # 执行矩阵乘法，将样本投影到新的低维空间。
    X_transformed = np.dot(X.cpu().numpy(), components)

    return X_transformed.reshape(X.shape[0], X.shape[1], n_components)  # 将返回的二维降维矩阵还原为三维形状

# 使用示例
data = np.random.rand(145, 145, 200)
n_components = 50
n_neighbors = 10

new_data = lle(data, n_components, n_neighbors)
print(new_data.shape)

