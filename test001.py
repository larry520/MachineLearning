from sklearn.datasets import make_moons
import numpy as np

data = make_moons(n_samples=1000, noise=0.4)

a = data[0]
b = data[1].reshape(-1,1)
c = np.concatenate((a,b),axis=1)

x = 2 * np.random.rand(100, 1)  # rand 随机(0~1) randn 正态分布随机数(-1~1)
y = 4 + 3 * x + np.random.randn(100, 1)  # 4*x0 + 3*x1 +random
x_b = np.concatenate((np.ones((100, 1)), x), axis=1)  # add x0=1 to each instance