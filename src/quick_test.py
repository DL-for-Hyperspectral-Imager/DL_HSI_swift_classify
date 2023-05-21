# added by Mangp, to run test quickly in this py
# 此文件将多次运行的输出写入对应的测试文件夹下，未作其他处理
import subprocess
import argparse

parser = argparse.ArgumentParser(description = 'Run the script multiple times with different arguments.')
# exp文件夹下你的文件夹名
parser.add_argument('--folder', type = str, help = 'Input file path', default = 'mangp')
args = parser.parse_args()

model = 'nearest'
preprocess_list = ['pca', 'ica', 'lda']
n_bands_list = [25, 50, 75, 100, 125, 150, 175]
# 要统计的量
data_list = ['Global accuracy', 'Kappa', 'Training time', 'Predicting time']

# 以不同的波段数和降维方法进行多次测试
for preprocess in preprocess_list:
    for n_bands in n_bands_list:
        # 输出文件为 model_preprocess_nbands.txt, 后续加入其他参数
        # 样例： nearest_pca_100.txt
        output_path = '../exp/' + args.folder + '/' + model + '_' + preprocess + '_' + \
                      str(n_bands) + '.txt'

        with open(output_path, 'w') as f:
            result = subprocess.run(
                    ['python', 'main.py', '--model', model, \
                     '--preprocess', preprocess, '--n_bands', str(n_bands)], \
                    capture_output = True)
            f.write(result.stdout.decode())
