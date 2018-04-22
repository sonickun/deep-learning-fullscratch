# coding: utf-8

import sys
import os
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

sys.path.append(os.pardir)


class MultiLayerNet:
    """
    全結合による多層ニューラルネットワーク

    Parameters
    ----------------------
    input_size          : 入力サイズ (MNISTの場合は784)
    hidden_size_list    : 隠れ層のニューロンの数のリスト (e.g. [100,100,100])
    output_size         : 出力サイズ (MNISTの場合は10)
    activation          : 'relu' or 'sigmoid'
    weight_init_std     : 重みの標準偏差を指定 (e.g. 0.01)
        'relu'または'he'を指定した際は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した際は「Xavierの初期値」を設定
    weight_decay_lambda : Weight Decay (L2ノルム) の強さ
    use_dropout         : Dropoutを使用するかどうか
    dropout_ration      : Dropoutの割合
    use_batchNorm      : Batch Normalizationを使用するかどうか
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0,
                 use_dropout=False, dropout_ration=0.5, use_batchNorm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchNorm = use_batchNorm
        self.params = {}

        # 重みの初期化
        self.__init_weight(weight_init_std)

        # レイヤの生成
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            # Batch Normalization の適用
            if self.use_batchNorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])

            # 活性化関数の設定
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

            # Dropout の適用
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        # ソフトマックス関数
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """ 重みの初期設定

        Parameters
        ---------------------
        weight_init_std     : 重みの標準偏差を指定 (e.g. 0.01)
            'relu'または'he'を指定した際は「Heの初期値」を設定
            'sigmoid'または'xavier'を指定した際は「Xavierの初期値」を設定
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # ReLUを使う場合に推奨される初期値
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrr(1.0 / all_size_list[idx - 1])  # Sigmoidを使う場合に推奨される初期値
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    # 認識（推論）
    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    # 重みパラメータに対する勾配を求める
    # x: 入力データ, t: 教師データ
    def loss(self, x, t, train_flg=False):
        """損失関数を求める
        引数のxは入力データ、tは教師ラベル
        """
        y = self.predict(x, train_flg)

        # Weight Decay
        # 重みの2乗ノルム(L2ノルム)を損失関数に加算する
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    # 認識精度を求める
    # x: 入力データ, t: 教師データ
    def accuracy(self, X, T):
        Y = self.predict(X, train_flg=False)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1:
            T = np.argmax(T, axis=1)

        accuracy = np.sum(Y == T) / float(X.shape[0])

        return accuracy

    # 重みパラメータに対する勾配を求める
    # X: 入力データ, T: 教師データ
    def numerical_gradient(self, X, T):
        loss_W = lambda W: self.loss(X, T, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

            if self.use_batchNorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])

        return grads

    # 誤差逆伝播法
    # x: 入力データ, t: 教師データ
    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            # Weight DecayのL2正則化項の微分(λW)を加算
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW\
                                    + self.weight_decay_lambda * self.params['W' + str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            # Batch Normalization のパラメータを更新
            if self.use_batchNorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads
