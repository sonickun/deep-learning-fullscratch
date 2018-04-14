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


