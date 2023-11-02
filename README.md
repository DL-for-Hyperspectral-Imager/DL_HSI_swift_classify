# DL_HSI_swift_classify

# 深度学习在高光谱像元分类上的敏捷化研究

本仓库致力于使用敏捷化深度学习方法对高光谱图像进行像元分类。目前已实现支持向量机（SVM）分类器，后续将增加神经网络、深度网络等分类模型。

## 项目简介

高光谱图像是一种具有丰富频谱信息的遥感图像，在很多领域有着广泛的应用，如农业、环境监测等。本项目旨在开发一套高效的像元分类方法，从而提高对高光谱图像的应用效果。
## 环境配置

本项目基于Python3.10开发，需要安装以下依赖包：


1. `colorama==0.4.6`：一个跨平台的库，用于在终端输出中添加颜色和样式。它使得在Windows和其他操作系统上的控制台输出具有一致性。
2. `filelock==3.12.0`：一个简单的库，用于在多个进程之间实现文件级别的锁定。这有助于防止多个进程同时读写同一个文件，避免数据损坏和竞争条件。
3. `Jinja2==3.1.2`：一个用于Python的模板引擎，允许你使用Python表达式和控制结构创建动态HTML、XML或其他标记类型的文档。
4. `joblib==1.2.0`：一个用于轻量级并行计算和缓存结果的库，常用于大型数据处理任务和科学计算。
5. `MarkupSafe==2.1.2`：一个实现XML/HTML/XHTML的转义的库，通常与Jinja2一起使用，以确保模板输出中的特殊字符安全处理。
6. `mpmath==1.3.0`：一个用于任意精度浮点数运算（如高精度浮点数、复数）的库。它支持多种数学函数和常见操作。
7. `networkx==3.1`：一个用于创建、操作和分析复杂网络结构（如图和多重图）的库。它提供了丰富的算法和绘图功能。
8. `numpy==1.24.2`：一个用于科学计算的库，提供了多维数组对象、矩阵运算和其他高级数学函数。
9. `Pillow==9.5.0`：Python Imaging Library (PIL)的一个分支，用于处理和操作图像文件。它支持多种图像格式，并提供了图像处理和绘图功能。
10. `scikit-learn==1.2.2`：一个用于机器学习的库，提供了广泛的监督和无监督学习算法、特征提取、模型选择和评估等功能。
11. `scipy==1.10.1`：一个用于科学和技术计算的库，提供了线性代数、优化、积分、插值等多种数学函数和算法。
12. `sympy==1.11.1`：一个用于符号计算的库，支持符号代数、微积分、方程求解、离散数学等功能。
13. `threadpoolctl==3.1.0`：一个库，用于控制Python中线程池的行为，特别是与NumPy、SciPy和scikit-learn等库中使用的线程池相关。
14. `torch==2.0.0`：PyTorch库，一个广泛使用的深度学习框架，提供了张量计算、自动微分和GPU加速功能，支持构建和训练各种神经网络模型。
15. `tqdm==4.65.0`：一个快速、可扩展的Python进度条库，用于在命令行或其他环境中显示循环的进度。它可以轻松地与常见的Python循环结构（如for循环）集成。
16. `typing_extensions==4.5.0`：一个库，提供了一些额外的类型提示和功能，用于支持Python的类型检查。这些类型提示和功能可能在将来的Python版本中成为标准，但目前可以通过`typing_extensions`库使用。


注意：包版本视实际情况而定，一下包版本为测试机使用的版本，其他版本未经过测试。
安装命令：
```commandline
pip install -r requirements.txt
```

## 代码结构与功能

主要代码结构如下：

```text
- main.py：主程序，实现SVM分类器训练与预测
- utils.py：工具函数，包括加载数据集、波段降维等操作
- preprocess.py：数据预处理函数，包括数据划分、训练集与测试集构建等
- model.py: 分类器实现，包括SVM、KNN与NN等
- datasets.py: 数据的加载等
- Datasets/ : 高光谱数据
```

## 实验过程与结果

### 数据集

本项目使用的数据集为`IndianPines`。数据经过归一化处理后，再使用PCA、ICA等方法降维。

### 训练与测试

默认采用30%的数据作为训练集，70%的数据作为测试集。使用若干种分类器进行训练与预测。

### 分类结果

输出分类报告与准确率。具体效果需根据实际运行结果查看。

## 后续计划

1. 实现神经网络分类模型
2. 实现深度网络分类模型
3. 对比各种分类模型的效果
4. 优化代码结构与功能

## 运行程序

该程序使用命令行参数来控制程序的行为。以下是可以使用的参数及其默认值：

- `--dataset`：数据集名称，默认为 `IndianPines`.
- `--training_rate`：训练集样本比例，默认为 `0.1`.
- `--preprocess`：数据预处理方法名称，默认为 `None`.可选`PCA` / `ICA` / `LDA` / `TSNE`.
- `--n_bands`：预处理降维的目标维数，默认为`50`.
- `--model`：分类模型名称，默认为 `SVM`.可选：`SVM` / `NEAREST` / `NN` / `CNN1D` / `CNN2D`.
- `--n_runs`：运行程序的次数，默认为 1.
- `--img_path`：生成的图片的存储路径，默认为 `./result` 文件夹.
- `--load_model`：选择是否加载已经村好的模型数据，默认`None`.
- `--patch_size`：2D-CNN的卷积窗口大小，默认为10.
- `--bsz`：神经网络模型单轮训练的批大小.

你可以使用以下命令来运行程序：

SVM模型

```commandline
python main.py --dataset IndianPines --preprocess PCA --n_bands 125 --model nn --n_runs 50
python main.py --dataset IndianPines --preprocess PCA --n_bands 75 --model CNN1D
python main.py --dataset IndianPines --preprocess ICA --model SVM
python3 main.py --preprocess PCA --model CNN2D 

find . -name "*.py" -print | xargs wc -l
```

在上面的示例中，我们将 `--n_runs` 参数设置为 3，`--dataset` 参数设置为 'IndianPines'，`--preprocess` 参数设置为 'ICA'，`--model` 参数设置为 'SVM'。你可以根据需要调整这些参数的值。

更多运行示例请参照[run_example.md](./manual/run_examples.md)。


在程序中，使用 `argparse` 模块来解析命令行参数，并将其存储在相应的变量中。你可以使用这些变量来控制程序的行为。例如，使用 `n_runs` 变量来指定程序的运行次数，使用 `dataset_name` 变量来指定数据集名称等等。

运行结束后，将输出分类报告与准确率。

如：
SVM报告
```text
Classification Report:
               precision    recall  f1-score   support

           1       1.00      0.29      0.44        14
           2       0.65      0.69      0.67       428
           3       0.74      0.51      0.61       249
           4       0.67      0.20      0.30        71
           5       0.87      0.88      0.87       145
           6       0.93      0.97      0.95       219
           7       1.00      0.75      0.86         8
           8       0.95      1.00      0.97       143
           9       0.75      0.50      0.60         6
          10       0.69      0.71      0.70       292
          11       0.70      0.85      0.77       737
          12       0.79      0.53      0.64       178
          13       1.00      0.98      0.99        61
          14       0.91      0.96      0.93       380
          15       0.80      0.58      0.67       116
          16       1.00      0.96      0.98        28

    accuracy                           0.77      3075
   macro avg       0.84      0.71      0.75      3075
weighted avg       0.78      0.77      0.77      3075

Accuracy:  0.7746341463414634

```
