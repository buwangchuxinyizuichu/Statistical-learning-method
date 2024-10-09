from copy import deepcopy
from collections import Counter

import numpy as np
from numpy.linalg import norm


def partition_sort(arr, k, key=lambda x: x):
    """
    以枢纽(位置k)为中心将数组划分为两部分, 枢纽左侧的元素不大于枢纽右侧的元素

    :param arr: 待划分数组
    :param k: 枢纽位置
    :param key: 比较方式
    :return: None
    """
    start, end = 0, len(arr) - 1
    assert 0 <= k <= end
    while True:
        i, j, pivot = start, end, deepcopy(arr[start])
        while i < j:
            while i < j:
                # 从右向左查找较小元素
                while i < j and key(pivot) <= key(arr[j]):
                    j -= 1
                if i == j:
                    break
                arr[i] = arr[j]
                i += 1
                # 从左向右查找较大元素
                while i < j and key(arr[i]) <= key(pivot):
                    i += 1
                if i == j:
                    break
                arr[j] = arr[i]
                j -= 1
        arr[i] = pivot

        if i == k:
            return
        elif i < k:
            start = i + 1
        else:
            end = i - 1


def maxheap_replace(heap: list[tuple], new_node: tuple, key=lambda x: x[1]):
    """
    大根堆替换堆顶元素

    :param heap: 大根堆/列表（包含的元素都是包含了 KDNode实例 和 当前样本点到这个实例的距离的元组）
    :param new_node: 应包含 新节点 和 当前样本点到这个新节点的距离
    :return: None
    """
    root, son, end = 0, 1, len(heap) - 1
    while son <= end:
        if son < end and key(heap[son]) < key(heap[son + 1]):  # 如果左子节点比右子节点小，那么son指向右子节点
            son += 1
        if key(heap[son]) <= key(new_node):
            break
        heap[root] = heap[son]
        root, son = son, (son << 1) + 1  # son先指向左子节点
    heap[root] = new_node


def maxheap_heappush(heap: list[tuple], new_node: tuple, key=lambda x: x[1]):
    """
    大根堆插入元素

    :param heap: 大根堆/列表（包含的元素都是包含了 KDNode实例 和 当前样本点到这个实例的距离的元组）
    :param new_node: 应包含 新节点 和 当前样本点到这个新节点的距离
    :return: None
    """
    heap.append(new_node)
    pos = len(heap) - 1
    while pos > 0:
        parent_pos = (pos - 1) >> 1
        if key(heap[parent_pos]) >= key(new_node):
            break
        heap[pos] = heap[parent_pos]
        pos = parent_pos
    heap[pos] = new_node


class KDNode:
    """kd树节点"""

    def __init__(self, data=None, label=None, left=None, right=None, axis=None, parent=None):
        """
        :param data: 数据
        :param label: 数据标签
        :param left: 左孩子节点
        :param right: 右孩子节点
        :param axis: 节点在kd树中的分割轴
        :param parent: 父节点
        """
        self.data = data
        self.label = label
        self.left = left
        self.right = right
        self.axis = axis
        self.parent = parent


class KDTree:
    """kd树"""

    def __init__(self, data, labels=None):
        """
        :param data: 输入特征集, n_samples*n_features
        :param labels: 输入标签集, 1*n_samples
        """
        self.root = None
        self.y_valid = False if labels is None else True
        self.create(data, labels)

    def create(self, data, labels=None):
        """
        构建kd树

        :param data: 输入特征集, n_samples*n_features
        :param labels: 输入标签集, 1*n_samples
        :return: KDNode
        """

        def build_tree(data: list, axis: int, parent=None):
            """
            递归生成kd树

            :param data: 合并标签后输入集
            :param axis: 切分轴
            :param parent: 父节点
            :return: KDNode
            """
            n_samples = np.shape(data)[0]
            if n_samples == 0:
                return None
            mid = n_samples >> 1
            partition_sort(data, mid, key=lambda x: x[axis])

            if self.y_valid:
                kd_node = KDNode(data[mid][:-1], data[mid][-1], axis=axis, parent=parent)
            else:
                kd_node = KDNode(data[mid], axis=axis, parent=parent)

            next_axis = (axis + 1) % dimensions
            kd_node.left = build_tree(data[:mid], next_axis, kd_node)
            kd_node.right = build_tree(data[mid + 1:], next_axis, kd_node)
            return kd_node

        print("building kd_tree...")
        dimensions = np.shape(data)[1]
        if self.y_valid:
            data = np.hstack((np.array(data), np.array([labels]).T)).tolist()  # 把每个标签对应放到每个样本的最后，这样在建立kd树的时候就可以把数据和对应的标签绑在一起处理
        self.root = build_tree(data, 0)

    def search_knn(self, point, k, dist=None):
        """
        kd树中搜索当前样本点的k个最近邻样本

        :param point: 样本点
        :param k: 近邻数
        :param dist: 度量方式
        :return:
        """

        def search(kd_node):
            """
            递归搜索k近邻节点

            :param kd_node: KDNode
            :return: None
            """
            if kd_node is None:
                return
            data = kd_node.data
            distance = p_dist(data)
            if len(max_heap) < k:
                maxheap_heappush(max_heap, (kd_node, distance))
            elif distance < max_heap[0][1]:
                maxheap_replace(max_heap, (kd_node, distance))

            axis = kd_node.axis
            if abs(point[axis] - data[axis]) < max_heap[0][1] or len(max_heap) < k:
                search(kd_node.left)
                search(kd_node.right)
            elif point[axis] < data[axis]:
                search(kd_node.left)
            else:
                search(kd_node.right)

        if self.root is None:
            return Exception('Error: kd-tree must not be null.')
        if k < 1:
            raise ValueError("Error: k must be greater than 0.")

        if dist is None:  # 默认使用2范数度量距离
            p_dist = lambda x: norm(np.array(x) - np.array(point))  # 这是一个简易的函数，x为输入向量
        else:
            p_dist = lambda x: dist(x, point)

        max_heap = []  # 存储了搜索过程中遇到的节点到当前点的距离的大根堆，最多存储搜索过程中前k个距离最小的节点
        search(self.root)
        return sorted(max_heap, key=lambda x: x[1])

    def search_nn(self, point, dist=None):
        """
        搜索当前样本点的最近邻样本

        :param point: 当前样本点
        :param dist: 度量方式
        :return:
        """
        return self.search_knn(point, 1, dist)[0]

    def pre_order(self, now=KDNode()):
        """先序遍历"""
        if now is None:
            return
        elif now.data is None:
            now = self.root

        yield now
        for son in self.pre_order(now.left):
            yield son
        for son in self.pre_order(now.right):
            yield son

    @classmethod  # 它表示下面的方法是一个类方法，类方法的第一个参数是类本身，通常命名为 cls
    def height(cls, root):
        """kd-tree深度"""
        if root is None:
            return 0
        else:
            return max(cls.height(root.left), cls.height(root.right)) + 1


class KNN_Classifier:
    """KNN分类器"""

    def __init__(self, k, dist=None):
        self.k = k
        self.dist = dist
        self.kd_tree = None

    def data_processing(self, X):
        """数据预处理"""
        X = np.array(X)
        self.x_min = np.min(X, axis=0)  # 对特征向量的每个特征来说，从所有样本的这个特征中取最大值
        self.x_max = np.max(X, axis=0)  # 对特征向量的每个特征来说，从所有样本的这个特征中取最大值
        X = (X - self.x_min) / (self.x_max - self.x_min)  # 对每个特征归一化，具体是原本每个样本的特征向量都减去一个self.x_min，然后都再除以一个self.x_max-self.x_min
        return X

    def build_kdtree(self, X, y):
        """建立kd树"""
        print('fitting...')
        X = self.data_processing(X)
        self.kd_tree = KDTree(X, y)

    def predict(self, X) -> list:
        """预测类别"""
        if self.kd_tree is None:
            raise Exception('Error: kd-tree must be built before prediction!')
        knn = lambda x: self.kd_tree.search_knn(point=x, k=self.k, dist=self.dist)
        y_pred = []
        X = (X - self.x_min) / (self.x_max - self.x_min)  # 在测试阶段使用训练阶段保存下来的预处理参数（如最大值和最小值等）对测试数据进行预处理
        for x in X:
            y = Counter(r[0].label for r in knn(x)).most_common(1)[0][0]
            y_pred.append(y)
        return y_pred

    def score(self, X, y):
        """评估准确率"""
        y_pred = self.predict(X)
        correct_num = len(np.where(np.array(y_pred) == np.array(y))[0])  # 两者相等的个数
        return correct_num / len(y)


def test():
    """模型测试"""
    X, y = [], []
    with open(r"./datasets/knn_dataset.txt") as f:
        for line in f:
            tmp = line.strip().split('\t')
            X.append(tmp[:-1])
            y.append(tmp[-1])
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    """训练误差"""
    knc = KNN_Classifier(10)
    knc.build_kdtree(X, y)
    print(knc.score(X, y))  # 0.963
    print()
    """测试误差"""
    X_train, X_test = X[:980], X[-20:]
    y_train, y_test = y[:980], y[-20:]
    knc = KNN_Classifier(10)
    knc.build_kdtree(X_train, y_train)
    print(knc.score(X_test, y_test))  # 1.0


if __name__ == '__main__':
    test()
