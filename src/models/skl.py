import sklearn

def train_svm(X_train, y_train):
    # 加载svm分类器
    # svm_classifier = sklearn.svm.SVC(kernel='rbf', C=10, gamma=0.001)
    svm_classifier = sklearn.svm.SVC(
        class_weight="balanced", kernel="rbf", C=10, gamma=0.001
    )
    svm_classifier.fit(X_train, y_train)
    return svm_classifier


def train_knn(X_train, y_train):
    # 加载knn分类器
    knn_classifier = sklearn.neighbors.KNeighborsClassifier()
    # 通过选择最佳的邻居数量来执行knn分类器的参数调整
    # verbose为0表示不输出训练进度和信息
    knn_classifier = sklearn.model_selection.GridSearchCV(
        knn_classifier,
        {
            "n_neighbors": [1, 3, 5, 7, 9, 10, 13, 15, 20],
            "metric": ["euclidean", "manhattan", "chebyshev"],
        },
        verbose=0,
        n_jobs=8,
    )
    knn_classifier.fit(X_train, y_train)
    print("The best k is %d:" % knn_classifier.best_params_["n_neighbors"])
    return knn_classifier
