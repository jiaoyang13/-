import numpy as np

# 马尔科夫前向算法
def forward_prob(model, Observe, States):

    A, B, pi = model
    N = States.size
    T = Observe.size

    alpha = pi * B[:, Observe[0]]
    #print('(1)计算初值alpha_1(i)', alpha)

    for t in range(0, T - 1):
        #print('t = ', t +1, 'alpha=', t + 1, 'i =', alpha)
        alpha = alpha.dot(A)*B[:, Observe[t + 1]]
    print('输出Prob：', alpha.sum())
    return alpha.sum()

# 马尔科夫后向算法
def backward_prob(model, Observe, States):
    A, B, pi = model
    N = States.size
    T = Observe.size

    beta = np.ones((N,))
    for t in range(T - 2, -1, -1):
        beta = A.dot(B[:, Observe[t + 1]] * beta)
    prob = pi.dot(beta * B[:, Observe[0]])
    print('prob=', prob)
    return prob


A = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ])

B = np.array([[0.5, 0.5],
            [0.4, 0.6],
            [0.7, 0.3]
            ])
pi = np.array([0.2, 0.4, 0.4])
model = (A, B, pi)
Observe = np.array([0, 1, 0])
States = np.array([1, 2, 3])
forward_prob(model, Observe, States)
backward_prob(model, Observe, States)
