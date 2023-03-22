import numpy as np


class Treestump:
    def __init__(self):
        return

    # 求解根据特征features的分类阈值、分类误差、分类结果
    def best_threshold(self, features, labels, weights):
        error = 100000.0  # 无穷大
        best_v = 0.0
        # 单维features
        n_step = 50  # 设定步数
        features_min = features.min()
        features_max = features.max()
        stepsize = (features_max - features_min) / n_step  # 计算步长
        direct, compare_array = None, None
        for i in range(n_step):
            v = features_min + stepsize * i
            if v not in features:
                # 误分类计算
                compare_array_positive = np.where(features > v, 1, -1)
                tarray = np.where(compare_array_positive == labels, 0, 1)
                weight_error_positive = np.inner(weights, tarray)
                if weight_error_positive < error or 1 - weight_error_positive < error:
                    if weight_error_positive < 0.5:
                        error = weight_error_positive
                        compare_array = compare_array_positive
                        direct = 1
                    else:
                        error = 1 - weight_error_positive
                        compare_array = compare_array_positive * -1
                        direct = -1
                    best_v = v
        return best_v, direct, error, compare_array

    def fit(self, train_data, train_label, weights):
        self.M, self.N = train_data.shape
        self.error = 10000
        # 根据特征维度, 选择误差最小的
        for j in range(self.N):
            features = train_data[:, j]
            # 分类阈值，分类误差，分类结果
            best_v, final_direct, best_error, compare_array = self.best_threshold(features, train_label, weights)
            if best_error < self.error:
                self.error = best_error
                self.v = best_v
                self.direct = final_direct
                self.clf_result = compare_array
                self.axis = j
            if self.error == 0:
                break

    def predict(self, test):
        return np.where(test[:, self.axis] > self.v, self.direct, -self.direct)
