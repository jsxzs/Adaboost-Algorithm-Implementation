import numpy as np
import time

target = np.genfromtxt('targets.csv')
base_list = [1, 5, 10, 100]

from adaboost import AdaBoost

test1 = AdaBoost(base=0)
time_start = time.time()
test1.fit("data.csv", "targets.csv")
time_end = time.time()
print('训练时间：', time_end - time_start, 's')
testy1 = test1.predict("data.csv")

for base_num in base_list:
    acc = []
    for i in range(1, 11):
        fold = np.genfromtxt('experiments/base%d_fold%d.csv' % (base_num, i), delimiter=',', dtype=int)
        accuracy = sum(target[fold[:, 0] - 1] == fold[:, 1]) / fold.shape[0]
        acc.append(accuracy)

    print(np.array(acc).mean())

print('预测准确率：', sum(target == testy1) / len(target))
