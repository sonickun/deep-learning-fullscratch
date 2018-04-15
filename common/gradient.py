# coding: utf-8
import numpy as np


# 勾配降下法
# パラメータxに関するfの偏微分（勾配）を返す
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001

    # xと同じ形のゼロ行列を作成
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    # 数値微分
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = float(tmp_val) - h
        fxh2 = f(x)  # f(x-h)
        # 中心差分を計算
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val  # 値をもとに戻す
        it.iternext()

    return grad

# Test
# def test_function(x):
#     return x[0]**2 + x[1]**2
#
# a = numerical_gradient(test_function, np.array([2.0, 4.0]))
#
# print(a)



