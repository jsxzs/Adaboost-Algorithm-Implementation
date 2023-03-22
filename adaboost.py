# encoding=utf8
import numpy as np
from treestump import Treestump
from logisticregression import LogisticRegression


# adaboost算法
class AdaBoost:
    def __init__(self, base=0):
        self.base_kind = base

    def init_args(self, datasets, labels):
        self.X = datasets
        self.Y = labels
        self.M, self.N = datasets.shape
        # 弱分类器数目和集合
        self.clf_sets = []
        # 初始化weights
        self.weights = np.array([1.0 / self.M] * self.M)
        # G(x)系数 alpha
        self.alpha = []

    # 计算alpha
    def _alpha(self, error):
        return 0.5 * np.log((1 - error) / error)

    # 规范化因子
    def _Z(self):
        s = sum(self.weights)
        self.weights /= s

    # 权值更新
    def _w(self, a, clf):
        self.weights *= np.exp(-1 * a * self.Y * clf)

    # # 计算训练集上的准确率
    # def accuracy(self, features, labels):
    #     result = np.zeros(self.M)
    #     for i in range(len(self.clf_sets)):
    #         ts = self.clf_sets[i]
    #         result += self.alpha[i] * ts.predict(features)
    #     clf_array = np.where(result > 0, 1, -1)
    #     return sum([1 for k in range(self.M) if clf_array[k] == labels[k]]) / self.M

    # Z-score标准化
    def standardscaler(self, x):
        for i in range(x.shape[1]):
            x[:, i] = (x[:, i] - self.features_mean[i]) / self.features_std[i]
        return x

    # Min-Max标准化
    def minmaxscaler(self, x):
        for i in range(x.shape[1]):
            p = self.features_max[i] - self.features_min[i]
            if p != 0:  x[:, i] = (x[:, i] - self.features_min[i]) / p
        return x

    # 训练模型
    def fit(self, x_file, y_file):
        data = np.genfromtxt(x_file, delimiter=',', dtype=np.float32)
        self.features_mean = np.mean(data, axis=0)
        self.features_std = np.std(data, axis=0)
        self.features_min = np.min(data, axis=0)
        self.features_max = np.max(data, axis=0)
        data = self.standardscaler(data)  # 数据标准化
        data = self.minmaxscaler(data)
        label = np.genfromtxt(y_file, delimiter=',', dtype=int)
        label = np.where(label == 0, -1, 1)  # 将标签为0的改为-1
        best_clf_sets, best_alpha, best_acc = None, None, -1
        # 4种基分类器数目
        base_list = [1, 5, 10, 100]
        for j in range(4):
            print('base_num=', base_list[j])
            self.base_num = base_list[j]
            kfold = 10
            kfold_size = len(data) // kfold
            # 10折交叉验证
            for i in range(kfold):
                foldindex = np.zeros(len(data))
                foldindex[i * kfold_size:(i + 1) * kfold_size] = 1
                trainindex = foldindex == 0
                testindex = foldindex == 1
                train_data = data[trainindex]  # 第i折的训练数据
                train_label = label[trainindex]  # 第i折的训练标签
                test_data = data[testindex]  # 第i折的测试数据
                test_label = label[testindex]  # 第i折的测试标签
                self.init_args(train_data, train_label)  # 初始化参数
                # 训练模型
                if self.base_kind == 1:  # 基分类器为决策树桩时
                    for epoch in range(self.base_num):
                        stump = Treestump()
                        stump.fit(train_data, train_label, self.weights)
                        # 若基分类器比随机猜测还差则中止算法
                        if stump.error >= 0.5:
                            break
                        # 计算G(x)系数a
                        a = self._alpha(stump.error)
                        self.alpha.append(a)
                        # 记录分类器
                        self.clf_sets.append(stump)
                        # 权值更新
                        self._w(a, stump.clf_result)
                        # 归一化权重
                        self._Z()
                else:  # 基分类器为逻辑回归时
                    for epoch in range(self.base_num):
                        lr = LogisticRegression(n_iters=500, learning_rate=10)
                        lr.fit(train_data, train_label, self.weights)
                        if 0.5 - lr.error < 0.001:
                            break
                        a = self._alpha(lr.error)
                        self.alpha.append(a)
                        self.clf_sets.append(lr)
                        self._w(a, lr.clf_result)
                        self._Z()
                # 输出10折测试集的预测结果
                out = self.test(test_data)
                acc = sum(np.where(out == test_label, 1, 0)) / len(out)
                out = np.insert(out.reshape(1, -1), 0, np.arange(i * kfold_size + 1, i * kfold_size + kfold_size + 1),
                                axis=0)
                out = [[row[j] for row in out] for j in range(kfold_size)]
                np.savetxt('experiments/base%d_fold%d.csv' % (self.base_num, i + 1), out, fmt='%d', delimiter=',')
                # 10折过程中，记录最好的模型
                if best_acc < acc:
                    best_acc = acc
                    best_clf_sets = self.clf_sets
                    best_alpha = self.alpha
        # 训练结束后，选择最好的模型作为预测测试数据的模型
        self.clf_sets = best_clf_sets
        self.alpha = best_alpha

    def test(self, test):
        l = len(test)
        result = np.zeros(l)
        for i in range(len(self.clf_sets)):
            t = self.clf_sets[i]
            result += self.alpha[i] * t.predict(test)
        ans = np.where(result > 0, 1, 0)
        return ans

    def predict(self, x_file):
        test_data = np.genfromtxt(x_file, delimiter=',', dtype=np.float32)
        test_data = self.standardscaler(test_data)
        test_data = self.minmaxscaler(test_data)
        return self.test(test_data)
