import numpy as np
from sklearn import preprocessing

import torch


class NeturaNet(torch.nn.Module):
    def __init__(self, n_f, n_h, n_o):
        super(NeturaNet, self).__init__()
        self.h = torch.nn.Linear(n_f, n_h)
        self.p = torch.nn.Linear(n_h, n_o)

    def forward(self, x):
        x = self.h(x)
        x = self.p(x)
        return x


class MultiVariableLinearRegression:

    def TargetFunction(self, x1, x2):
        w1, w2, b = 2, 5, 10
        return w1 * (20 - x1) + w2 * x2 + b

    def CreateSampleData(self, m):
        X = np.zeros((m, 2))
        X[:, 0:1] = (np.random.random(m) * 20 + 2).reshape(m, 1)
        X[:, 1:2] = np.random.randint(40, 120, (m, 1))
        Y = self.TargetFunction(X[:, 0:1], X[:, 1:2])
        Noise = np.random.randint(1, 100, (m, 1)) - 50
        Y = Y + Noise
        return X, Y

    def Normalize(self, X, Y):
        min_max_scaler = preprocessing.MinMaxScaler()
        X_minMax = min_max_scaler.fit_transform(X)
        Y_minMax = min_max_scaler.fit_transform(Y)
        return min_max_scaler, X_minMax, Y_minMax

    def ReNormalize(self, min_max_scaler, Y):
        Y_minMax = min_max_scaler.inverse_transform(Y)
        return Y_minMax

    def Train(self):
        X, Y = self.CreateSampleData(1000)
        sacler, X1, Y1 = self.Normalize(X, Y)
        X1 = X1.astype(np.float32)
        Y1 = Y1.astype(np.float32)

        xt = torch.from_numpy(X1)
        yt = torch.from_numpy(Y1)

        net = NeturaNet(2, 3, 1)

        optimizer = torch.optim.SGD(net.parameters(), 0.1)
        loss_f = torch.nn.MSELoss()
        pre = None
        for t in range(10000):
            pre = net(xt)
            loss = loss_f(pre, yt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Y2 = pre.data.numpy()
        yP = self.ReNormalize(sacler, Y2)
        print(Y.transpose() - yP.transpose())


if __name__ == '__main__':
    s = MultiVariableLinearRegression()
    s.Train()
