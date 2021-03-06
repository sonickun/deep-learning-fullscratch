# coding: utf-8
# 二層ニューラルネットワークの学習

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from network.two_layer_net import TwoLayerNet
from common.optimizer import *

sys.path.append(os.pardir)


def main():
    # データの読み込み
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # ニューラルネットワークの構成を定義
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    iters_num = 10000  # 繰り返しの回数を適宜設定
    train_size = x_train.shape[0]
    batch_size = 100

    optimizer = Adam()

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 1エポックあたりの繰り返し数
    iter_per_epoch = max(train_size / batch_size, 1)
    print("Iteration per Epoch:", iter_per_epoch)

    for i in range(iters_num):
        # ミニバッチの取得
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 勾配の計算
        # grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)  # 高速化（誤差逆伝播法）

        # パラメータの更新
        optimizer.update(network.params, grad)

        # 学習経過の記録
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        # print("Loss:", loss)

        # 1エポックごとに認識精度を計算
        if i % iter_per_epoch == 0:
            train_acc = network.accuarcy(x_train, t_train)
            test_acc = network.accuarcy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("[+] Epoch:\t%d\ttrain acc, test acc |\t%f,\t%f"
                  % (i/iter_per_epoch, train_acc, test_acc))

    # グラフの描画
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc', marker=markers['train'])
    plt.plot(x, test_acc_list, label='test acc', linestyle='--', marker=markers['test'])
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()


