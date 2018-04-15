# coding: utf-8
# ニューラルネットワークの推論処理

import sys
import os
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

sys.path.append(os.pardir)


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


def main():
    # 学習データと正解ラベルの読み込み
    x, t = get_data()

    # 学習済みパラメータの読み込み
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)  # 最も確率の高い要素のインデックスを取得
        print("[+] Predict: %d, Label:%d, Result: " % (p, t[i]), end='')
        if p == t[i]:
            accuracy_cnt += 1
            print("OK")
        else:
            print("NG")

    # 正答率の出力
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


if __name__ == '__main__':
    main()


