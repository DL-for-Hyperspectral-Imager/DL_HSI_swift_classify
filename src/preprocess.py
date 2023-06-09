# -*- coding: utf-8 -*-


import numpy as np
#import cupy as cp

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from cuml.manifold import LocallyLinearEmbedding
#from sklearn.manifold import LocallyLinearEmbedding


def preprocess(hsi_img, gt, preprocess_name, n_bands):
    # 归一化
    img = np.asarray(hsi_img, dtype = np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    preprocess_name = preprocess_name.lower()
    if n_bands == 0:
        return img
    if preprocess_name == 'pca':
        img = pca_sklearn(img, n_bands)
    elif preprocess_name == 'ica':
        img = ica_sklearn(img, n_bands)
    elif preprocess_name == 'lda':
        img = lda_sklearn(img, gt, n_bands)  ##
    #elif preprocess_name == 'lle':
        #img = lle_gpu(img, n_bands)
    elif preprocess_name == 'tsne':
        img = tsne_sklearn(img, n_bands)
    return img


def pca_numpy(img, k):
    # img: (H, W, C)
    # k: 降维后的维度
    # return: (H, W, k)
    height, width, channel = img.shape
    img = np.transpose(img, (2, 0, 1))
    img = np.reshape(img, (img.shape[0], -1))
    img = img - np.mean(img, axis = 1, keepdims = True)
    cov = np.dot(img, img.T)
    eig_val, eig_vec = np.linalg.eig(cov)
    eig_val = np.real(eig_val)
    eig_vec = np.real(eig_vec)
    eig_val = np.sort(eig_val)
    eig_vec = eig_vec[:, np.argsort(eig_val)]
    eig_vec = eig_vec[:, -k:]
    img = np.dot(eig_vec.T, img)
    img = np.reshape(img, (k, img.shape[1]))
    img = np.transpose(img, (1, 0))
    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
    return img


def pca_sklearn(data, k):
    # 原始数据的形状
    m, n, p = data.shape
    # 将数据reshape成(m*n,p)的形式
    data_reshape = np.reshape(data, (m * n, p))
    # 创建PCA对象
    pca = PCA(n_components = k)
    # pca = PCA(0.95)
    # 对数据进行降维
    data_pca = pca.fit_transform(data_reshape)
    # 将数据reshape回原来的形状
    data_pca_reshape = np.reshape(data_pca, (m, n, data_pca.shape[-1]))
    # 返回降维后的数据
    return data_pca_reshape


def ica_sklearn(img, k):
    # 将三维图像数据reshape成N*(height*width*channel)的矩阵
    X = np.reshape(img, (-1, img.shape[2]))

    # ICA降维，保留前k个独立成分
    ica = FastICA(n_components = k, random_state = 0, whiten = 'unit-variance')
    X_ica = ica.fit_transform(X)

    # 将降维后的数据reshape回原来的三维图像形状
    img_ica = np.reshape(X_ica, (img.shape[0], img.shape[1], k))

    return img_ica


def lda_sklearn(img, gt, k):
    # 原始数据的形状
    m, n, p = img.shape
    # 将数据reshape成(m*n,p)的形式,有监督降维算法,所以要提供gt.
    X = np.reshape(img, (m * n, p))
    y = gt.reshape(m * n)
    # 创建lda对象,该算法要求 n_components不能大于原始数据维度和类别数
    # 经测试发现在kvm分类器下取13为最优
    lda = LinearDiscriminantAnalysis(n_components = k)  ##
    # 对数据进行降维
    lda.fit(X, y)
    X_lda = lda.transform(X)
    # 将数据reshape回原来的形状
    X_lda_reshape = np.reshape(X_lda, (m, n, X_lda.shape[-1]))
    # 返回降维后的数据
    return X_lda_reshape


# def lle_sklearn(img, k):
#     img = cp.asarray(img)
#     # 原始数据的形状
#     m, n, p = img.shape
#     # 将数据reshape成(m*n,p)的形式,无监督降维算法
#     X = np.reshape(img, (m * n, p))
#     #X = cp.asarray(X)
#     # 创建lle对象
#     lle = LocallyLinearEmbedding(n_components=k, eigen_solver='dense', neighbors_algorithm = 'kd_tree')
#     # 对数据进行降维
#     X_lle = lle.fit_transform(X)
#     #X_lle = cp.asnumpy(X_lle)
#     # 将数据reshape回原来的形状
#     X_lle_reshape = np.reshape(X_lle, (m, n, X_lle.shape[-1]))
#     X_lle_reshape = cp.asnumpy(X_lle_reshape)
#     # 返回降维后的数据
#     return X_lle_reshape



# def lle_gpu(img, k):
#     # 原始数据的形状
#     m, n, p = img.shape
#     # 将数据reshape成(m*n,p)的形式,无监督降维算法
#     X = np.reshape(img, (m * n, p))
#     #将数据转移到GPU内存
#     X_gpu = cp.asarray(X)
#     # 创建lle对象
#     lle = LocallyLinearEmbedding(n_components=k, eigen_solver='dense', neighbors_algorithm='kd_tree')
#     # 对数据进行降维
#     X_lle_gpu = lle.fit_transform(X_gpu)
#     # 将数据reshape回原来的形状，并转换回CPU的numpy数组格式
#     X_lle_reshape = np.reshape(cp.asnumpy(X_lle_gpu.get()), (m, n, k))
#     # 返回降维后的数据
#     return X_lle_reshape


def tsne_sklearn(data, k):
    # 原始数据的形状.
    m, n, p = data.shape
    # 将数据reshape成(m*n,p)的形式
    data_reshape = np.reshape(data, (m * n, p))
    # pca = PCA(n_components=50)
    # data_pca = pca.fit_transform(data_reshape)
    # 创建tsne对象
    tsne = TSNE(n_components=3, method='barnes_hut', random_state=0)
    # 对数据进行降维
    data_tsne = tsne.fit_transform(data_reshape)
    # 将数据reshape回原来的形状
    data_tsne_reshape = np.reshape(data_tsne, (m, n, data_tsne.shape[-1]))
    # 返回降维后的数据
    return data_tsne_reshape

