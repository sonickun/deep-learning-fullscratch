# coding: utf-8

import sys
import os
from common.optimizer import *

sys.path.append(os.pardir)


class Trainer:
    """ニューラルネットの訓練を行うクラス
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr': 0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True):
        """
        :param network: ニューラルネットワークの定義
        :param x_train: 訓練データ
        :param t_train: 訓練ラベル
        :param x_test: テストデータ
        :param t_test: テストラベル
        :param epochs: エポック回数
        :param mini_batch_size: ミニバッチのサイズ
        :param optimizer: パラメータ更新の最適化手法
        :param optimizer_param: optimizerのパラメータ
        :param evaluate_sample_num_per_epoch:
        :param verbose: 学習経過の表示の可否
        """
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # Optimizer
        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'nesterov': Nesterov,
                                'addgrad': AdaGrad, 'rmsprpo': RMSprop, 'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]
        # 1エポックあたりの繰り返し回数
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        # ミニバッチの取得
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # 勾配の計算（誤差逆伝播法）
        grads = self.network.gradient(x_batch, t_batch)

        # パラメータの更新
        self.optimizer.update(self.network.params, grads)

        # 学習経過の記録
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        # if self.verbose:
        #     print("[+] Iters: %d, Epoch: %d, Train Loss: %f" % (self.current_iter, self.current_epoch, loss))

        # 1エポックごとに精度を計算
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if self.evaluate_sample_num_per_epoch is not None:
                # 使用するデータの数を制限する場合
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print("=== Epoch: %d, Train acc: %f, test acc: %f ===" % (self.current_epoch, train_acc, test_acc))

        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("Test acc:", test_acc)




