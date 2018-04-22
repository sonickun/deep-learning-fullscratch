# coding: utf-8

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from network.multi_layer_net import MultiLayerNet
from common.trainer import Trainer


sys.path.append(os.pardir)

def main():
    # データの読み
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

    use_dropout = True
    dropout_ratio = 0.2

    # ニューラルネットワークの定義
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio)

    # 学習器の定義
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=301, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True)

    # 学習開始
    trainer.train()

    train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

    # グラフの描画=======
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, marker=markers['train'], label='train', markevery=10)
    plt.plot(x, test_acc_list, marker=markers['test'], label='test', markevery=10)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()

