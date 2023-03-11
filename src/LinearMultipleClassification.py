import numpy as np

import torch

import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn import preprocessing


class NeturaNet(torch.nn.Module):
    def __init__(self, n_f, n_h, n_o):
        super(NeturaNet, self).__init__()
        self.h = torch.nn.Linear(n_f, n_h)
        self.p = torch.nn.Linear(n_h, n_o)

    def forward(self, x):
        x = self.h(x)
        x = self.p(x)
        x = F.log_softmax(x)
        return x


class LinearMultipleClassification:

    def fun1(self, x):
        y = 2.5 * x - 10
        return y

    def fun2(self, x):
        y = 0.3 * x + 5
        return y

    def fun3(self, x):
        y = - 1 * x + 10
        return y

    def CreateSampleData(self, m):
        X = np.random.random((m, 2)) * 10
        Y = np.zeros((m, 1))
        for i in range(m):
            x1 = X[i, 0]
            x2 = X[i, 1]
            y1 = self.fun1(x1)
            y2 = self.fun2(x1)
            y3 = self.fun3(x1)

            noise = (np.random.rand() - 0.5) * 2
            x2 = abs(x2 - noise)
            if x2 > y1 and x2 > y2 and x2 > y3:
                Y[i, 0] = 1
            elif x2 > y1 and x2 < y2 and x2 < y3:
                Y[i, 0] = 2
            elif x2 < y1 and x2 < y2 and x2 > y3:
                Y[i, 0] = 3
            else:
                Y[i, 0] = 0

        return X, Y

    def Normalize(self, X):
        min_max_scaler = preprocessing.MinMaxScaler()
        X_minMax = min_max_scaler.fit_transform(X)
        return X_minMax

    def Train(self):
        m = 500
        X, Y = self.CreateSampleData(m)

        xn = self.Normalize(X)

        xt = torch.from_numpy(xn).type(torch.FloatTensor)
        yt = torch.from_numpy(Y.reshape(-1)).type(torch.LongTensor)

        net = NeturaNet(2, 4, 4)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        loss_f = torch.nn.NLLLoss()
        pre = None
        for t in range(10000):
            pre = net(xt)
            loss = loss_f(pre, yt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Yp = torch.max(pre.data, dim=1)[1].numpy()
        Yp = np.array(np.split(Yp, m))
        self.Draw(m, xn, Y)
        self.Draw(m, xn, Yp)
        print("end")

    def Draw(self, m, X, Y):

        for i in range(m):
            if Y[i, 0] == 1:
                plt.plot(X[i, 0], X[i, 1], '.', c='r')
            elif Y[i, 0] == 2:
                plt.plot(X[i, 0], X[i, 1], 'x', c='g')
            elif Y[i, 0] == 3:
                plt.plot(X[i, 0], X[i, 1], 'o', c='b')
            else:
                plt.plot(X[i, 0], X[i, 1], 'h', c='y')

        plt.show()
        print('end')


if __name__ == '__main__':
    s = LinearMultipleClassification()
    s.Train()
