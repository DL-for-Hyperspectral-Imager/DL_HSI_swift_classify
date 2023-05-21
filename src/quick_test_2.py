# added by Mangp, to run test quickly in this py
# 此文件将各次运行结果全部保存起来了，方便作图分析等
import subprocess
import argparse
import main
import matplotlib.pyplot as plt
import numpy as np

model_list = ['cnn2d']  # 'nn'
preprocess_list = ['pca']
n_bands_list = [25, 50, 75, 100, 125, 150, 175, 200]

for model in model_list:
    # 以不同的波段数和降维方法进行多次测试
    for preprocess in preprocess_list:
        train_times = []
        predict_times = []
        accuracys = []
        for n_bands in n_bands_list:
            # run_results, Training_time, Predicting_time即为本次运行的结果
            # run_results中包含了accuracy, F1 score by class, confusion matrix,为字典
            hyperparams = {}
            hyperparams = {'dataset':'IndianPines',
                           'n_runs':1,
                           'training_rate':0.3,
                           'preprocess': preprocess,
                           'n_bands':n_bands,
                           'model':model,
                           'img_path':'result',
                           'load_model':None}
            if(model == 'cnn1d' or model == 'cnn2d'):
                hyperparams['n_runs'] = 200
                hyperparams['patch_size'] = 10
                hyperparams['bsz']        = 1000

            run_results, Training_time, Predicting_time = main.main(
                    show_results_switch = False, hyperparams = hyperparams)
            # 记录数据，可以增加其他属性
            accuracys.append(run_results['Accuracy'])
            train_times.append(Training_time)
            predict_times.append(Predicting_time)
        # 绘图
        fig = plt.figure(figsize = (10, 4))
        plt.subplot(121)
        plt.title("                                    For model of " + model + " and preprocess of " + preprocess)
        plt.xlabel('bands')
        plt.ylabel('accuracy/%')
        plt.plot(n_bands_list, accuracys, "og-")
        plt.subplot(122)
        plt.xlabel('bands')
        plt.ylabel('time/s')
        plt.plot(n_bands_list, train_times, "ob-", label = "train time")
        plt.plot(n_bands_list, predict_times, "or-", label = "predict time")
        plt.legend()
        plt.savefig("../result/" + model + '_' + preprocess + '/' + model + "_" + preprocess + ".jpg")
        # plt.show()


preprocess = 'lda'  # lda需要用不同的bands

n_bands_list = list(np.arange(1, 17))
for model in model_list:
    train_times = []
    predict_times = []
    accuracys = []
    for n_bands in n_bands_list:
        # run_results, Training_time, Predicting_time即为本次运行的结果
        # run_results中包含了accuracy, F1 score by class, confusion matrix,为字典
        hyperparams = {}
        hyperparams = {'dataset':'IndianPines',
                        'n_runs':1,
                        'training_rate':0.1,
                        'preprocess': preprocess,
                        'n_bands':n_bands,
                        'model':model,
                        'img_path':'result',
                        'load_model':None}
        if(model == 'cnn1d' or model == 'cnn2d'):
                hyperparams['n_runs'] = 200
                hyperparams['patch_size'] = 10
                hyperparams['bsz']        = 1000
        run_results, Training_time, Predicting_time = main.main(
                show_results_switch = False, hyperparams = hyperparams)
        # 记录数据，可以增加其他属性
        accuracys.append(run_results['Accuracy'])
        train_times.append(Training_time)
        predict_times.append(Predicting_time)
    # 绘图
    fig = plt.figure(figsize = (10, 4))
    plt.subplot(121)
    plt.title("                                    For model of " + model + " and preprocess of " + preprocess)
    plt.xlabel('bands')
    plt.ylabel('accuracy/%')
    plt.plot(n_bands_list, accuracys, "og-")
    plt.subplot(122)
    plt.xlabel('bands')
    plt.ylabel('time/s')
    plt.plot(n_bands_list, train_times, "ob-", label = "train time")
    plt.plot(n_bands_list, predict_times, "or-", label = "predict time")
    plt.legend()
    plt.savefig("../result/" + model + '_' + preprocess + '/' + model + "_" + preprocess + ".jpg")
    # plt.show()