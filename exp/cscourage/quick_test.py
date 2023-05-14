#added by Mangp, to run test quickly in this py
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Run the script multiple times with different arguments.')
#exp文件夹下你的文件夹名
parser.add_argument('--folder', type=str, help='Input file path', default='mangp')

model = 'nn'
#model_list = ['svm', 'nearest', 'nn']
preprocess = 'lda'
#preprocess_list = ['pca', 'ica', 'lda']
n_bands_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

args = parser.parse_args()

# for preprocess in preprocess_list:
#     for n_bands in n_bands_list:
#         #输出文件为 model_preprocess_nbands.txt, 后续加入其他参数
#         #样例： nearest_pca_100.txt
#         output_path = 'exp/'+args.folder + '/' + model + '_' + preprocess + '_' + str(n_bands) + '.txt'
#         print(output_path)

#         with open(output_path, 'w')as f:
#             result = subprocess.run(['python', 'main.py', '--model', model,\
#                                       '--preprocess', preprocess,'--n_bands', str(n_bands)], capture_output=True)
#             f.write(result.stdout.decode())
#for model in model_list:
for n_bands in n_bands_list:
    output_path = 'exp/'+args.folder + '/' + model + '_' + preprocess + '_' + str(n_bands) + '.txt'
    print(output_path)

    with open(output_path, 'w')as f:
        result = subprocess.run(['python', 'main.py', '--model', model,\
                                  '--preprocess', preprocess,'--n_bands', str(n_bands)], capture_output=True)
        f.write(result.stdout.decode())
        
            
            
            
            
            
            