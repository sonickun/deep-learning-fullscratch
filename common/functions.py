# coding: utf-8
import numpy as np


def indentity_function(x):
    return x


# ステップ関数
def step_function(x):
    return np.array(x > 0, dtype=np.int)


# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU関数
def relu(x):
    return np.maximum(0, x)


# ソフトマックス関数
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


# 交差エントロピー誤差
# y: ニューラルネットワークの出力
# t: 教師データ
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# Test
# y = np.array([[0.0, 0.4, 0.6], [0.2, 0.1, 0.7]])
# t = np.array([[0, 1, 0], [0, 0, 1]])
# print(cross_entropy_error(y, t))













