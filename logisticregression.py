import numpy as np


# 对数几率回归基分类器
class LogisticRegression:
    def __init__(self, n_iters=1000, learning_rate=1.0):
        self.n_iters = n_iters
        self.learning_rate = learning_rate

    def init_args(self, datasets):
        self.M, self.N = datasets.shape
        self.w = np.ones(self.N)

    # sigmoid函数
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    # 训练函数
    def fit(self, train_data, train_label, weights):
        train_data = np.insert(train_data, 0, 1, 1)  # 在数据集的第0列加一列1，进行模型转换
        self.init_args(train_data)  # 初始化参数
        label = np.where(train_label == -1, 0, 1)
        for i in range(self.n_iters):
            h = self.sigmoid(train_data.dot(self.w.T))
            gradient = (weights.T * (h - label.T)).T.dot(train_data)
            if np.linalg.norm(gradient) <= 0.01: break  # 若梯度的模小于0.01，结束训练
            self.w -= self.learning_rate * gradient
        # 计算误分率
        self.clf_result = self.sigmoid(train_data.dot(self.w.T))
        self.clf_result = np.where(self.clf_result >= 0.5, 1, -1)
        self.error = np.inner(weights, np.where(self.clf_result != train_label, 1, 0))
        # 若误分率大于0.5，正负反转
        if self.error > 0.5:
            self.clf_result *= -1
            self.error = 1 - self.error

    def predict(self, test_data):
        test_data = np.insert(test_data, 0, 1, 1)
        ans = self.sigmoid(test_data.dot(self.w.T))
        return np.where(ans >= 0.5, 1, -1)
