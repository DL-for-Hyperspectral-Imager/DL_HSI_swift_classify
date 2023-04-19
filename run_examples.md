## 运行示例
pca + svm
```commandline
python main.py --dataset IndianPines --n_runs 1 --sample_rate 0.3 --preprocess PCA --model SVM
```
result:
```text
Classification Report:
               precision    recall  f1-score   support

           1       1.00      0.00      0.00        32
           2       1.00      0.00      0.00      1000
           3       1.00      0.00      0.00       581
           4       1.00      0.00      0.00       166
           5       1.00      0.00      0.00       338
           6       0.41      0.84      0.55       511
           7       1.00      0.00      0.00        20
           8       1.00      0.00      0.00       335
           9       1.00      0.00      0.00        14
          10       1.00      0.00      0.00       680
          11       0.35      0.99      0.52      1719
          12       1.00      0.00      0.00       415
          13       1.00      0.00      0.00       143
          14       0.69      0.99      0.81       886
          15       1.00      0.00      0.00       270
          16       1.00      0.00      0.00        65

    accuracy                           0.42      7175
   macro avg       0.90      0.18      0.12      7175
weighted avg       0.76      0.42      0.26      7175

Accuracy:  0.4190940766550523
```