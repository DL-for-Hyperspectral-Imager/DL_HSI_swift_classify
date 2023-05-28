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
import time
import subprocess
import argparse
import main
import matplotlib.pyplot as plt
import numpy as np
import os

model_list = ['svm', 'knn', 'nn', 'cnn1d', 'cnn2d']  # 'nn'
preprocess_list = ['pca', 'ica', 'lda', 'tsne']
n_bands_list_normal = [0, 25, 50, 75, 100, 125, 150, 175, 200]  # 0 代表不降维， 以比较不降维和降维的效果
n_bands_list_lda = list(np.arange(2, 17))
res_folder = "result"
print('model_list:', model_list)
print('preprocess_list:', preprocess_list)
allcnts = len(model_list) * len(preprocess_list) * len(n_bands_list_normal)
cnt = 0
Start = time.time()
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
            cnt = cnt + 1
            Curr = time.time()
            print("\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n")
            print("No %d / %d, during %.1f min" % (cnt, allcnts, (Curr - Start) / 60))
            # run_results, Training_time, Predicting_time即为本次运行的结果
            # run_results中包含了accuracy, F1 score by class, confusion matrix,为字典
            hyperparams = {
                    'dataset'      : 'IndianPines',
                    'n_runs'       : 1,
                    'training_rate': 0.3,
                    'preprocess'   : preprocess,
                    'n_bands'      : n_bands,
                    'model'        : model,
                    'res_folder'   : res_folder,
                    'load_model'   : False,
                    'patch_size'   : 0,
                    'batch_size'   : 0}
            if (model == 'cnn1d' or model == 'cnn2d' or model == 'nn'):
                hyperparams['n_runs'] = 200
                hyperparams['patch_size'] = 10
                hyperparams['batch_size'] = 1000
            try:
                run_results = main.main(
                        show_results_switch = False,
                        hyperparams = hyperparams)
                # 记录数据，可以增加其他属性
                accuracys.append(run_results['accuracy'])
                preprocess_times.append(run_results['preprocess_time'])
                train_times.append(run_results['training_time'])
                predict_times.append(run_results['predicting_time'])
            except Exception as e:
                print(e)
        # 绘图
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))

        axes[0].plot(n_bands_list, accuracys, "og-")
        axes[0].set_xlabel('Bands', fontsize = 14)
        axes[0].set_ylabel('Accuracy (%)', fontsize = 14)
        axes[1].plot(n_bands_list, train_times, "ob-", label = "train time")
        axes[1].plot(n_bands_list, predict_times, "or-", label = "predict time")
        axes[1].plot(n_bands_list, preprocess_times, "oy-", label = "preprocess time")
        axes[1].set_xlabel('Bands', fontsize = 14)
        axes[1].set_ylabel('Time (s)', fontsize = 14)

        fig.suptitle("Preprocess %s, Model %s" % (preprocess, model), fontsize = 16)
        plt.legend()
        plt.savefig(
                os.path.join(
                        os.getcwd(), "..", res_folder, "%s_%s" % (preprocess, model),
                                                       "%s_%s.png" % (preprocess, model)))
        # plt.show()

        # 创建一个新的文件，如果文件已经存在则删除它，保证每次重新运行时是覆写而不是追加
        filepath = os.path.join(
                os.getcwd(), "..", res_folder, "%s_%s" % (preprocess, model), "%s_%s.txt" % (preprocess, model))
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
