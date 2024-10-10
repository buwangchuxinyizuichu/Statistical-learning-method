import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class NaoveBayes_Model:

    def __init__(self, X_train: pd.DataFrame, Y_train: pd.DataFrame):
        self.X_train = X_train
        self.Y_train = Y_train
        self.p_y = {1: np.mean(Y_train.values), 0: 1 - np.mean(Y_train.values)}

    def cal_P(self, data, feature, value):
        num = len(data[data[feature] == value])
        return (num + 1) / (len(data) + len(np.unique(self.X_train[feature] + data[feature])))

    def predict(self, X_test: pd.DataFrame) -> list:
        self.predictions = []
        data_labels0 = self.X_train[self.Y_train.values == 0]  # 类别为0的数据，其余数据取为NaN
        data_labels1 = self.X_train[self.Y_train.values == 1]  # 类别为1的数据，其余数据取为NaN
        for i in range(len(X_test)):
            x = X_test.iloc[i]  # 取第i个样本
            P_feature_label0 = 1  # 初始化P(特征|类别=0)
            P_feature_label1 = 1  # 初始化P(特征|类别=1)
            P_feature = 1  # 初始化P(特征)
            for feature in X_test.columns:
                # 计算P(特征|类别)
                P_featurei_label0 = self.cal_P(data_labels0, feature, x[feature])  # 计算P(训练样本中特征=测试样本的第i个特征|训练样本中类别=0)
                if P_featurei_label0 == 0:
                    P_featurei_label0 = 1 / len(data_labels0)
                P_feature_label0 *= self.cal_P(data_labels0, feature, x[feature])
                P_feature_label1 *= self.cal_P(data_labels1, feature, x[feature])  # 计算P(训练样本中特征=测试样本的第i个特征|训练样本中类别=1)
                # 计算P(特征)
                P_feature *= self.cal_P(self.X_train, feature, x[feature])
            P_0 = (P_feature_label0 * self.p_y[0]) / P_feature  # 计算测试样本的P(类别=0)
            P_1 = (P_feature_label1 * self.p_y[1]) / P_feature  # 计算测试样本的P(类别=1)
            self.predictions.append([1 if P_1 >= P_0 else 0])
        return self.predictions


def test():
    X, Y = load_iris(return_X_y=True)
    X_train, X_test, Y_train, Y_test = train_test_split(pd.DataFrame(X), pd.DataFrame(Y), test_size=0.3)
    model = NaoveBayes_Model(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print(accuracy_score(Y_pred, Y_test))


if __name__ == '__main__':
    test()
