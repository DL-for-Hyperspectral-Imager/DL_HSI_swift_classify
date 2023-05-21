
preprocess_list = ['pca', 'ica']
n_bands_list = [25, 50, 75, 100, 125, 150, 175, 200]

for model in model_list:
    # 以不同的波段数和降维方法进行多次测试
    for preprocess in preprocess_list:
        train_times = []
        predict_times = []
        accuracys = []