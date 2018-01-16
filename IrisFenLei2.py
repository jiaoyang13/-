#BP多层神经网络为两种花分类

import numpy as np
from sklearn.datasets import load_iris

dataSets = load_iris()
X = np.array(dataSets.data) #花的叶片数据
Y = np.array(dataSets.target) #花的种类数据
Y = Y.reshape(1, 150)

X = X[0 : 75, :] #一共150条数据四种花，取其中两种花的属性数据
Y = Y[:, 0 : 75] #只取两种花的分类

X1 = np.ones(X.shape[0])
X = np.insert(X, 0, values=X1, axis= 1) #把1插入到属性数据的第0列，为了后面的偏置值

#权值初始化，三层神经网络，隐层权值为V， 输出层权值为W
V = np.random.random((5, 3))*2 - 1
W = np.random.random((3, 1))*2 - 1

#学习率
lr = 0.11

#激活函数
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def dsigmoid(x):
    return x*(1-x)

def update():
    global X, Y, W, V, lr

    #print('w.shape=', W.shape, 'V=', V.shape)

    L1 = sigmoid(np.dot(X, V))  #隐层输出
    L2 = sigmoid(np.dot(L1, W))  #输出层输出
    #print('L2=', L2.shape, 'L1=', L1.shape)
    #print('Y=', Y.shape)
    L2_delta = (Y.T - L2) * dsigmoid(L2)
    L1_delta = L2_delta.dot(W.T)* dsigmoid(L1)
    #print('L2_delta.shape=', L2_delta.shape)
    #print('L1_delta.shape=', L1_delta.shape)

    W_C = lr * L1.T.dot(L2_delta)
    V_C = lr * X.T.dot(L1_delta)

    #print('W_C=', W_C.shape, 'V_C=', V_C.shape)

    W = W + W_C
    V = V + V_C

for i in range(10000):
    update() #更新权值

    if i % 500 == 0:
        L1 = sigmoid(np.dot(X, V))  #隐层输出
        L2 = sigmoid(np.dot(L1, W))  #输出层输出
        print('Y=', Y)
        print('L2=', L2)
        print('Error:', np.mean(np.abs(Y.T - L2)))
