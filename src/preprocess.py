# -*- coding: utf-8 -*-


import numpy as np
# import cupy as cp

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from cuml.manifold import LocallyLinearEmbedding
# from sklearn.manifold import LocallyLinearEmbedding

PREPROCESS = ["PCA", "ICA", "LDA", "LLE", "TSNE"]


def preprocess(
    hsi_img: np.ndarray, gt: np.ndarray, name: str, n_bands: int
) -> np.ndarray:
    """
    Return:
        img: img after preprocessed, from original bands to n_bands
            numpy.ndarray, shape:(h, w, n_bands)
    """
    # 归一化
    img = np.asarray(hsi_img, dtype=np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    name = name.upper()

    if (name not in PREPROCESS) or (n_bands == 0):
        # n_bands == 0, no dims reduction
        return img

    print(
        f"* Start {name} preprocessing..., from {hsi_img.shape[2]} -> {n_bands} bands"
    )
    if name == "PCA":
        img = pca_sklearn(img, n_bands)
    elif name == "ICA":
        img = ica_sklearn(img, n_bands)
    elif name == "LDA":
        img = lda_sklearn(img, gt, n_bands)  
    elif name == "LLE":
        # img = lle_gpu(img, n_bands)
        pass
    elif name == "TSNE":
        img = tsne_sklearn(img, n_bands)

    print("* Preprocessing finished!")
    print("--- After preprocessing, dataset shape:", img.shape)
    return img



def pca_sklearn(img, k):
    h, w, b = img.shape

    pca = PCA(n_components=k)
    # PCA = PCA(0.95)

    img_pca = pca.fit_transform(np.reshape(img, (h * w, b)))

    # reshape回原来的形状
    return np.reshape(img_pca, (h, w, -1))


def ica_sklearn(img, k):
    h, w, b = img.shape

    # ICA降维，保留前k个独立成分
    ica = FastICA(n_components=k, random_state=0, whiten="unit-variance")

    img_ica = ica.fit_transform(np.reshape(img, (h * w, b)))

    return np.reshape(img_ica, (h, w, -1))


def lda_sklearn(img, gt, k):
    h, w, b = img.shape
    # lda 要求 n_components不能大于原始数据维度和类别数
    # 经测试发现在kvm分类器下取13为最优
    lda = LinearDiscriminantAnalysis(n_components=k)

    # reshape成(m*n,p)的形式,有监督降维算法,所以要提供gt.
    img_lca = lda.fit_transform(np.reshape(img, (h * w, b)), gt.reshape(-1))

    return np.reshape(img_lca, (h, w, -1))


def tsne_sklearn(img, k):
    h, w, b = img.shape

    lle = TSNE(n_components=3, method="barnes_hut", random_state=0)

    img_tsne = lle.fit_transform(np.reshape(img, (h * w, b)))

    return np.reshape(img_tsne, (h, w, -1))


# def lle_sklearn(img, k):
#     img = cp.asarray(img)
#     # 原始数据的形状
#     m, n, p = img.shape
#     # 将数据reshape成(m*n,p)的形式,无监督降维算法
#     X = np.reshape(img, (m * n, p))
#     # X = cp.asarray(X)
#     # 创建lle对象
#     lle = LocallyLinearEmbedding(
#         n_components=k, eigen_solver="dense", neighbors_algorithm="kd_tree"
#     )
#     # 对数据进行降维
#     X_lle = lle.fit_transform(X)
#     # X_lle = cp.asnumpy(X_lle)
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

# def get_decomposition_model(process):
#     if process == "PCA":
#         pass
#     elif process == "ICA":
#         img = ica_sklearn(img, n_bands)
#     elif process == "LDA":
#         img = lda_sklearn(img, gt, n_bands)  ##
#     elif process == "LLE":
#         # img = lle_gpu(img, n_bands)
#         pass
#     elif process == "TSNE":
#         pass


def pca_numpy(img, k):
    # img: (H, W, C)
    # k: 降维后的维度
    # return: (H, W, k)
    height, width, channel = img.shape
    img = np.transpose(img, (2, 0, 1))
    img = np.reshape(img, (img.shape[0], -1))
    img = img - np.mean(img, axis=1, keepdims=True)
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