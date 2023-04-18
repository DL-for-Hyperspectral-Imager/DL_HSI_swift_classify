
import numpy as np

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

def preprocess(img, preprocess_name):
    if preprocess_name == 'PCA':
        img = pca_sklearn(img, 50)
    elif preprocess_name == 'ICA':
        img = ica_sklearn(img, 50)
    return img


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

def pca_sklearn(data, k):
    # 原始数据的形状
    m, n, p = data.shape
    # 将数据reshape成(m*n,p)的形式
    data_reshape = np.reshape(data, (m*n, p))
    # 创建PCA对象
    pca = PCA(n_components=k)
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
    ica = FastICA(n_components=k, random_state=0, whiten='unit-variance')
    X_ica = ica.fit_transform(X)

    # 将降维后的数据reshape回原来的三维图像形状
    img_ica = np.reshape(X_ica, (img.shape[0], img.shape[1], k))

    return img_ica