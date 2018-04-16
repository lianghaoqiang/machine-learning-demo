import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import cholesky
import random
import math

# 产生服从两个高斯分布的样本，使用em算法估算GMM模型参数
# (2PI)^(d/2) * |sigma|^(-1/2) exp{-1/2(x-mu).T * sigma^-1 * (x-mu)}

# 样本数量
sample_num1 = 500
sample_num2 = 1000

# 产生服从两个分布的样本
mu = np.array([[1,5], [6,6]])
Sigma1 = np.array([[1, 0], [0, 1]])
R1 = cholesky(Sigma1)
s1 = np.dot(np.random.randn(sample_num1, 2), R1) + mu[0,:]
Sigma2 = np.array([[1,0],[0,2]])
R2 = cholesky(Sigma2)
s2 = np.dot(np.random.randn(sample_num2, 2), R2) + mu[1,:]

x = np.append(s1,s2,axis=0)
plt.figure(figsize=[15,15])
plt.plot(s1[:,0],s1[:,1],'*')
plt.plot(s2[:,0],s2[:,1],'+')
plt.show()


# 初始化GMM模型参数，即Pi，mu,sigma
# x为样本点，k为GMM模型中分布个数,本例k=2
def init(x,k=2):
    n,d = np.shape(x)
    pi = np.array([1.0/k]*k,dtype=np.float)
    # 随机选取k个样本为各个分布的mu
    mu = np.array(random.sample(list(x),k))
    # 样本协方差为各个分布的sigma
    sigma = np.zeros((d,d,k))
    for i in range(k):
        sigma[:,:,i] = np.cov(x.T)
    return pi,mu,sigma


def em(x,k=2,iter_num = 200):
    n, d = np.shape(x)
    pi, mu, sigma = init(x,k)

    cost = 0
    iteration = 0
    while iteration < iter_num:
        # E step: 计算每个样本属于各个分布的概率
        # theta(i,j) = pi_k*N(xi|mu_j,sigma_j)/sum(分子j从1到k)
        P = np.zeros([n, k], dtype=np.float)
        for j in range(k):
            sigma_j = sigma[:, :, j]
            x_ = np.mat(x-mu[j,:])
            for i in range(n):
                tmp = math.pow(2*math.pi, d/2)*np.linalg.det(sigma_j)
                tmp2 = -1/2 * x_[i, :] * np.mat(sigma_j).getI() * x_[i, :].T
                P[i,j] = 1.0/tmp * np.exp(tmp2)

        theta = np.mat(np.array(P)*np.array(pi) )     # n*k
        theta = theta/np.sum(theta,axis=1)

        # M step：
        Nk = np.sum(theta,axis=0)    # 1*2
        mu = np.diagflat(1.0/Nk)*theta.T*x
        pi = Nk/n
        sigma = np.zeros([d,d,k],dtype=np.float)
        for j in range(k):
            x_ = x-mu[j, :]
            for i in range(n):
                sigma_tmp = x_[i,:].T*x_[i,:]*theta[i,j]/Nk[0,j]
                sigma[:, :, j] = sigma[:, :, j]+sigma_tmp

        labels = theta.argmax(axis=1)
        iteration = iteration + 1
        cost_tmp = sum(np.log(np.mat(P) * np.mat(pi).T))
        esp = abs(cost_tmp - cost)
        if esp<0.00001:
            break
        cost = cost_tmp

        print('iter %d  esp %.15lf' % (iteration, esp))

        if iteration % 1 == 0:
            plt.figure(figsize=[15, 15])
            for i in range(n):
                if labels[i] == 0:
                    plt.plot(x[i, 0], x[i, 1], 'go')
                else:
                    plt.plot(x[i, 0], x[i, 1], 'ro')
            plt.show()

    print('Sigma:', sigma)
    print('mu', mu)
    print('PI:', pi)

em(x)