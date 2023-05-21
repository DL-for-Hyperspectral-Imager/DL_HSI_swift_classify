# added by Mangp, to run test quickly in this py
# 此文件将各次运行结果全部保存起来了，方便作图分析等
import sys
import matplotlib

if sys.platform.startswith('win'):
    print('This is Windows')
elif sys.platform.startswith('linux'):
    matplotlib.use('Agg')
    print('This is Linux')
    print('using agg as backend')

import subprocess
import argparse
import main
import matplotlib.pyplot as plt
import numpy as np
import os

model_list = ['cnn2d']  # 'nn'
preprocess_list = ['pca']
n_bands_list_normal = [25, 50, 75, 100, 125, 150, 175, 200]
n_bands_list_lda = list(np.arange(1, 17))

for model in model_list:
    # 以不同的波段数和降维方法进行多次测试
    for preprocess in preprocess_list:
        preprocess_times = []
        train_times = []
        predict_times = []
        accuracys = []

        if preprocess == 'lda':
            n_bands_list = n_bands_list_lda
        else:
            n_bands_list = n_bands_list_normal
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
                           'load_model':None, 
                           'patch_size':None,
                           'bsz':None}
            if(model == 'cnn1d' or model == 'cnn2d'):
                hyperparams['n_runs'] = 200
                hyperparams['patch_size'] = 10
                hyperparams['bsz']        = 1000

            run_results = main.main(show_results_switch = False, 
                                    hyperparams = hyperparams)
            # 记录数据，可以增加其他属性
            accuracys.append(run_results['Accuracy'])
            preprocess_times.append(run_results['preprocess_time'])
            train_times.append(run_results['training_time'])
            predict_times.append(run_results['predicting_time'])
        # 绘图
        fig = plt.figure(figsize = (10, 4))
        plt.subplot(121)
        plt.title("                                                  For model of " + model + " and preprocess of " + preprocess)
        plt.xlabel('bands')
        plt.ylabel('accuracy/%')
        plt.plot(n_bands_list, accuracys, "og-")
        plt.subplot(122)
        plt.xlabel('bands')
        plt.ylabel('time/s')
        plt.plot(n_bands_list, train_times, "ob-", label = "train time")
        plt.plot(n_bands_list, predict_times, "or-", label = "predict time")
        plt.plot(n_bands_list, preprocess_times, "oy-", label = "preprocess time")
        plt.legend()
        plt.savefig(r"../result/" + model + "_" + preprocess + r"/" + model + "_" + preprocess + ".jpg")
        # plt.show()

        # 创建一个新的文件，如果文件已经存在则删除它，保证每次重新运行时是覆写而不是追加
        filepath = "../result/" + model + '_' + preprocess + '/' + model + "_" + preprocess + ".txt"
        if os.path.exists(filepath):
            os.remove(filepath)

        # 进行单次运行，追加数据到文件中
        with open(filepath, "a") as f:
            f.write("n_bands: \n")
            for bands in n_bands_list:
                f.write("%s\t    " % bands)
            f.write("\n")
            
            f.write("accuracys: \n")
            for ac in accuracys:
                f.write("%.4s  \t" % ac)
            f.write("\n")
            
            f.write("train_times: \n")
            for train in train_times:
                f.write("%.6s\t" % train)
            f.write("\n")
            
            f.write("predict_times: \n")
            for pred in predict_times:
                f.write("%.6s\t" % pred)
            f.write("\n")

            f.write("preprocess_times: \n")
            for prepro in preprocess_times:
                f.write("%.6s\t" % prepro)
            f.write("\n")
