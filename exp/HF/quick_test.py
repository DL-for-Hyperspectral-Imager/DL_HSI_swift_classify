#corrected by HF, to run test quickly in this py
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Run the script multiple times with different arguments.')
#exp文件夹下你的文件夹名
parser.add_argument('--folder', type=str, help='Input file path', default='HF')
modellist = ['svm','nearest','nn']

preprocess_list = ['ica',  'pca',  'lda']
n_bands_list = [25, 50, 75, 100, 125, 150, 175]

args = parser.parse_args()
for model in modellist:
    for preprocess in preprocess_list:
        for n_bands in n_bands_list:
            #输出文件为 model_preprocess_nbands.txt, 后续加入其他参数
            #样例： nearest_pca_100.txt
            output_path = './exp/'+args.folder + '/' + model + '_' + preprocess + '_' + str(n_bands) + '.txt'
            img_path = './exp/'+args.folder
            print(output_path)
    
            with open(output_path, 'w')as f:
                result = subprocess.run(['python', 'main.py', '--model', model,\
                                          '--preprocess', preprocess,'--n_bands', str(n_bands),'--img_path',img_path], capture_output=True
                                        )
                f.write(result.stdout.decode())
