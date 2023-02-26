import numpy as np
import torch
import matplotlib.pyplot as plt


class NeturaNet(torch.nn.Module):
    def __init__(self, n_f, n_h, n_o):
        super(NeturaNet, self).__init__()
        self.h = torch.nn.Linear(n_f, n_h)
        self.p = torch.nn.Linear(n_h, n_o)

    def forward(self, x):
        x = self.h(x)
        x = self.p(x)
        return x


class SingleVariableLinearRegression:
    def TargetFunction(self, X):
        noise = np.random.normal(0, 0.2, X.shape)
        W = 2
        B = 3
        Y = np.dot(X, W) + B + noise
        return Y

    def CreateSampleData(self, m):
        X = np.random.random((m, 1))
        Y = self.TargetFunction(X)
        return X, Y

    def Train(self):
        X, Y = self.CreateSampleData(1000)
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        xt = torch.from_numpy(X)

        yt = torch.from_numpy(Y)

        net = NeturaNet(1, 2, 1)

        optimizer = torch.optim.SGD(net.parameters(), 0.1)
        loss_f = torch.nn.MSELoss()
        pre = None
        for t in range(100):
            pre = net(xt)
            loss = loss_f(pre, yt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.Draw(X, Y, pre.data.numpy())
        print("end")

    def Draw(self, x, y, p):
        plt.ion()  # 画图
        plt.show()
        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, p, 'r-', 5)
        plt.pause(0.1)


if __name__ == '__main__':
    s = SingleVariableLinearRegression()
    s.Train()
