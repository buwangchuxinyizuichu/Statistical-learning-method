import numpy as np


class PerceptionMethod:

    def __init__(self, X: np.array, Y: np.array, learning_rate=0.1):
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Error: X and Y must be same when axis=0")
        else:
            self.X = X
            self.Y = Y
            self.learning_rate = learning_rate

    def train_online(self):  # 原始形式感知机（在线学习）
        try:
            weight = np.zeros(self.X.shape[1])
        except:
            raise ValueError("Error: X must be 2D array")
        b = 0
        epoch = 0  # 记录训练次数
        flag = 1  # 训练的时候分类有错误时为 1
        while flag:
            flag = 0
            for i in range(self.X.shape[0]):
                if self.Y[i] * (weight @ self.X[i] + b) <= 0:
                    weight += self.learning_rate * self.Y[i] * self.X[i]
                    b += self.learning_rate * self.Y[i]
                    epoch += 1
                    flag = 1
                    break  # 找出第一个错误后跳出循环进入下一轮训练使用的是：在线学习策略
        print(f"Epoch: {epoch}")
        print("Done!")
        return weight, b

    def train_offline(self):  # 原始形式感知机（离线学习）
        try:
            weight = np.zeros(self.X.shape[1])
        except:
            raise ValueError("Error: X must be 2D array")
        b = 0
        epoch = 0  # 记录训练次数
        while True:
            delta_weight = np.zeros(self.X.shape[1])  # 计算损失函数的梯度值
            delta_b = 0
            for i in range(self.X.shape[0]):
                if self.Y[i] * (weight @ self.X[i] + b) <= 0:
                    delta_weight += self.learning_rate * self.Y[i] * self.X[i]
                    delta_b += self.learning_rate * self.Y[i]
            if delta_weight.any():
                weight += delta_weight
                b += delta_b
                epoch += 1
            else:
                break
        print(f"Epoch: {epoch}")
        print("Done!")
        return weight, b

    def dual_train_online(self):  # 对偶形式感知机（在线学习）
        Gram = self.X @ self.X.T  # Gram[i]记录了第i个训练样本分别与所有训练样本之间的内积，矩阵之间的 @ 运算就是矩阵乘法
        alpha = np.zeros(self.X.shape[0])  # 每个训练样本在权重向量中的系数
        b = 0
        flag = 1  # 训练的时候分类有错误时为 1
        while flag:
            flag = 0
            for i in range(self.X.shape[0]):
                if self.Y[i] * (alpha * self.Y @ Gram[i] + b) <= 0:  # 矩阵之间的 * 运算是两个矩阵对应元素相乘
                    alpha[i] += self.learning_rate
                    b += self.learning_rate * self.Y[i]
                    flag = 1
                    break
        weight = alpha * self.Y @ self.X
        print("Done!")
        return weight, b
