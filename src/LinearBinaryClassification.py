import numpy as np

import torch
import matplotlib.pyplot as plt

import torch.nn.functional as F


class NeturaNet(torch.nn.Module):
    def __init__(self, n_f, n_h, n_o):
        super(NeturaNet, self).__init__()
        self.h = torch.nn.Linear(n_f, n_h)
        self.p = torch.nn.Linear(n_h, n_o)

    def forward(self, x):
        x = self.h(x)
        x = self.p(x)
        x = F.sigmoid(x)
        return x


class LinearBinaryClassification:
    def TargetFunction(slef, x1, x2):
        y = 2 * x1 - 0.5
        if x2 > y:
            return 1
        else:
            return 0

    def CreateSampleData(self, m):

        X = np.random.random((m, 2))
        Y = np.zeros((m, 1))
        for i in range(m):
            y = self.TargetFunction(X[i, 0], X[i, 1])
            Y[i, 0] = y
        return X, Y

    def Train(self):
        X, Y = self.CreateSampleData(500)
        xt = torch.from_numpy(X).type(torch.FloatTensor)
        yt = torch.from_numpy(Y).type(torch.FloatTensor)

        net = NeturaNet(2, 3, 1)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        loss_f = torch.nn.BCELoss()
        pre = None
        for t in range(5000):
            pre = net(xt)
            loss = loss_f(pre, yt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.Draw(500, X, Y, pre.data.numpy())
        print("end")

    def Draw(self, m, X, Y1, Y2):

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 4.5))
        for i in range(m):
            if Y1[i, 0] == 1:
                ax1.scatter(X[i, 0], X[i, 1], marker='x', c='g')
            else:
                ax1.scatter(X[i, 0], X[i, 1], marker='o', c='r')

        for i in range(m):
            if Y2[i, 0] >= 0.5:
                ax2.scatter(X[i, 0], X[i, 1], marker='x', c='g')
            else:
                ax2.scatter(X[i, 0], X[i, 1], marker='o', c='r')
        plt.show()


if __name__ == '__main__':
    s = LinearBinaryClassification()
    s.Train()
