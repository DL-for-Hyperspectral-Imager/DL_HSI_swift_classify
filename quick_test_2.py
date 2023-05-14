# added by Mangp, to run test quickly in this py
# 此文件将各次运行结果全部保存起来了，方便作图分析等
import subprocess
import argparse
import main

model = 'nearest'
preprocess_list = ['pca', 'ica', 'lda']
n_bands_list    = [25, 50, 75, 100, 125, 150, 175]

# 以不同的波段数和降维方法进行多次测试
for preprocess in preprocess_list:
    for n_bands in n_bands_list:
        # run_results, Training_time, Predicting_time即为本次运行的结果
        # run_results中包含了accuracy, F1 score by class, confusion matrix,为字典
        run_results, Training_time, Predicting_time = main.main(False, n_runs = 1, dataset = 'IndianPines', preprocess = preprocess, model = model, training_sample = 0.1,\
                                 n_bands = n_bands)
        # 下面进行分析处理...