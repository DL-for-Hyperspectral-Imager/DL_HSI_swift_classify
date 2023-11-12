# -*- coding: utf-8 -*-

import os
import joblib
from tqdm import trange

import numpy as np

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch.optim import Optimizer, AdamW
# local modules

from utils import plot_loss_acc

from dataset import ImgDataset

from models.fnn import FNN
from models.cnn import CNN1D, CNN2D
from models.skl import train_knn, train_svm

pwd = os.getcwd()

SKLEARN_MODEL = ["SVM", "KNN"]
TORCH_MODEL = ["NN", "CNN1D", "CNN2D"]


def train(
    hparams: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device="cpu",
    *,
    clf_folder="classifier",  # 将训练好的分类器保存在classifier文件夹下面
):
    """
    用于训练的函数
    """
    print("* Start training...")

    model_name = hparams["model"].upper()
    n_runs = hparams["n_runs"]

    sklearn_clf_folder = os.path.join(pwd, "..", clf_folder)
    torch_model_folder = os.path.join(pwd, "..", clf_folder, "torch_model", model_name)

    os.makedirs(sklearn_clf_folder, exist_ok=True)
    os.makedirs(torch_model_folder, exist_ok=True)

    if model_name in SKLEARN_MODEL:
        if model_name == "SVM":
            clf = train_svm(X_train, y_train)
        elif model_name == "KNN":
            clf = train_knn(X_train, y_train)
        else:
            raise Exception("undefined model name!")
        joblib.dump(clf, os.path.join(sklearn_clf_folder, model_name + "_" + "clf"))
        print("* Training finished!")
        return clf
    
    elif model_name in TORCH_MODEL:
        save_path = os.path.join(
            torch_model_folder,
            model_name + "_" + "runs" + str(n_runs) + ".pth",
        )

        batch_size = hparams["batch_size"]  # batch_size

        # move to device (cpu/gpu)
        model = get_model(model_name, hparams).to(device)

        train_dset = ImgDataset(X_train, y_train)
        test_dset = ImgDataset(X_test, y_test)

        train_iter = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
        test_iter = DataLoader(test_dset, batch_size)

        loss_record, acc_record = train_net(
            model, train_iter, test_iter, n_runs, 0.01, device
        )
        # plot_loss_acc(loss_record, acc_record)
        print("* Training finished!")

        return model, (loss_record, acc_record)
        # torch.save(model.state_dict(), save_path)


def train_net(
    net: Module,
    train_iter: DataLoader,
    test_iter: DataLoader,
    epochs: int,
    lr: float,
    device: str | torch.device,
):
    print("training on", device)
    net.to(device)

    optimizer: Optimizer = AdamW(net.parameters(), lr=lr, weight_decay=0.01)

    loss = CrossEntropyLoss(reduction="mean")

    loss_record = []
    acc_record = []

    t = trange(epochs, desc="epochs")
    for epoch in t:
        for batch_X, batch_y in train_iter:
            net.train()  # faster
            # batch_X: (batch_size, features)
            # batch_y: (batch_size, )
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            # batch_y_out: probability of each classes
            # shape(batch_size, n_classes)
            batch_y_out = net(batch_X)
            l: torch.Tensor = loss(batch_y_out, batch_y)  # torch.tensor(2.xx)
            l.backward()  # backward calculate gradient
            optimizer.step()  # update the weight
            optimizer.zero_grad()
        test_acc = eval_acc(net, test_iter, device)

        loss_record.append(l.item())
        acc_record.append(test_acc)

        t.set_postfix(loss=l.item(), testacc=test_acc, lr=lr)
    return loss_record, acc_record


def get_model(model_name, hparams):
    n_bands = hparams["t_bands"]
    n_classes = hparams["n_classes"]

    if model_name == "NN":
        model = FNN(n_bands, n_classes, dropout=True, p=0.5)
    elif model_name == "CNN1D":
        model = CNN1D(n_bands, n_classes)
    elif model_name == "CNN2D":
        model = CNN2D(n_bands, n_classes, patch_size=hparams["patch_size"])
    else:
        raise KeyError(f"{model_name} model is unknown.")

    return model


def predict(
    model_name: str,
    clf,
    X_test: np.ndarray,
    device=torch.device("cpu"),
) -> np.ndarray:
    print("* Start predicting...")

    model_name = model_name.upper()

    if model_name in SKLEARN_MODEL:
        y_pred = clf.predict(X_test)

    elif model_name in TORCH_MODEL:
        X_test = torch.tensor(X_test).to(device)
        clf.to(device)
        clf.eval()
        with torch.no_grad():
            y_pred = clf(X_test)
            y_pred = torch.topk(y_pred, k=1).indices
            y_pred = y_pred.cpu().numpy()
    else:
        print("The model name is wrong")
        y_pred = None
        
    print("* Predicting finished!")
    return y_pred


def cal_correct(y_hat, y) -> int:
    """Return correct predicted num."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return cmp.sum().item()


def eval_acc(net, data_iter, device=None) -> float:
    if isinstance(net, Module):
        net.eval()
    ccnt, acnt = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            ccnt += cal_correct(net(X), y)
            acnt += len(y)
    return ccnt / acnt
