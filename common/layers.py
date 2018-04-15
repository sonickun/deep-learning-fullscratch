# coding: utf-8

import numpy as np
from common.functions import *


# Reluレイヤ
class Relu:
    def __init__(self):
        self.mask = None

    # 順伝播
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    # 逆伝播
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


# Sigmoidレイヤ
class Sigmoid:
    def __init__(self):
        self.out = None

    # 順伝播
    def forward(self, x):
        out = sigmoid(x)
        self.out = out

        return out

    # 逆伝播
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


# Affineレイヤ
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    # 順伝播
    def forward(self, x):
        # テンソル（４次元のデータ）対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    # 逆伝播
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        # 入力データの形状に戻す（テンソル対応）
        dx = dx.reshape(*self.original_x_shape)

        return dx


# Softmax-with-Lossレイヤ
# →Softmaxレイヤ＋Cross Entropy Errorレイヤ
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # Softmaxの出力
        self.t = None  # 教師データ

    # 順伝播
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    # 逆伝播
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx






