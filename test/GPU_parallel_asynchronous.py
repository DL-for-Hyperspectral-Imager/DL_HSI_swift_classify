# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:58:57 2023

@author: Cou'ra'ge
"""

import numpy as np
import torch

torch.backends.cuda.max_split_size_mb = 1024


def lle(X, n_components, n_neighbors):
    """
    X:输入的样本，形式是N*D, N是样本数，D是样本特征维度
    n_components:表示降维到的维度数
    n_neighbors:表示LLE算法计算权重时的近邻数
    """
    X = X.reshape(-1, X.shape[-1])
    N, D = X.shape

    with torch.cuda.device(0):
        # 创建新的CUDA异步事件
        event = torch.cuda.Event(enable_timing=True)

        # 异步进程的计算，传输和分配内存
        X_ = torch.tensor(X, dtype=torch.float32, device='cuda', requires_grad=True)
        X_square = torch.sum(X_ ** 2, dim = 1)
        dists = X_square.reshape(-1, 1) + X_square.reshape(1, -1) - 2 * torch.matmul(X_, X_.t())

        # 计算完成之前记录事件的状态
        event.record()

        # 对于每个样本，找到它的k个近邻，并计算权重矩阵
        W = np.zeros((N, n_neighbors), dtype=np.float32)
        indices = np.zeros((N, n_neighbors), dtype=np.int32)
        for i in range(N):
            dists_i = dists[i]
            indices_i = dists_i.argsort()[1:n_neighbors+1]
            indices[i] = indices_i
            Xi = X[indices_i] - X[i]
            G = torch.matmul(Xi, Xi.t())
            G += 1e-3 * torch.eye(n_neighbors).cuda()
            w = torch.linalg.solve(G, torch.ones(n_neighbors).cuda())
            w /= w.sum()
            W[i,:] = w.cpu().numpy()

        # 等待计算完成
        event.synchronize()

        # 计算降维权重矩阵并进行矩阵乘法
        M = np.identity(N) - np.dot(W, W.T)
        eigenvalues, eigenvectors = np.linalg.eig(np.dot(M.T, M))
        indices = np.argsort(eigenvalues)[: n_components]
        components = eigenvectors[:, indices]
        X_transformed = np.dot(X, components)

    return X_transformed.reshape(X.shape[0], X.shape[1], n_components)

# 使用示例
data = np.random.rand(145, 145, 200)
n_components = 50
n_neighbors = 10

new_data = lle(data, n_components, n_neighbors)
print(new_data.shape)
