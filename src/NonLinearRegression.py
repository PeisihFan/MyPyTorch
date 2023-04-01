import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch

import torch.nn.functional as F
class NeturaNet(torch.nn.Module):
    def __init__(self, n_f, n_h, n_o):
        super(NeturaNet, self).__init__()
        self.z1 = torch.nn.Linear(n_f, n_h)
        self.z2 = torch.nn.Linear(n_h, n_h)
        self.z3 = torch.nn.Linear(n_h, n_o)
        self.s = torch.nn.ReLU()
    def forward(self, x):
        x = self.z1(x)
        x = self.s(x)
        x = self.z3(x)
        return x
class NonLinearRegression:
    def TargetFunction(self,x):
        p1 = 0.4 * (x ** 2)
        p2 = 0.3 * x * np.sin(15 * x)
        p3 = 0.01 * np.cos(50 * x)
        y = p1 + p2 + p3 - 0.3
        return y

    def CreateSampleData(self,num_train, num_test):
        # create train data
        x1 = np.random.random((num_train, 1))
        y1 = self.TargetFunction(x1) + (np.random.random((num_train, 1)) - 0.5) / 10


        # create test data
        x2 = np.linspace(0, 1, num_test).reshape(num_test, 1)
        y2 = self.TargetFunction(x2)

        return x1,y1,x2,y2

    def draw(self,x1,y1,x2,y2):
        plt.scatter(x1, y1, s=1, c='b')
        plt.scatter(x2, y2, s=4, c='r')
        plt.show()

    def Normalize(self, X):
        min_max_scaler = preprocessing.MinMaxScaler()
        X_minMax = min_max_scaler.fit_transform(X)
        return min_max_scaler, X_minMax

    def Train(self):
        x1,y1,x2,y2 = self.CreateSampleData(1000,200)

        xs,x1s= self.Normalize(x1)
        ys, y1s= self.Normalize(y1)
        xt = torch.from_numpy(x1s).float()
        yt = torch.from_numpy(y1s).float()



        net = NeturaNet(1, 10, 1)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
        loss_f = torch.nn.MSELoss()
        pre = None
        for t in range(10000):
            pre = net(xt)
            loss = loss_f(pre, yt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        x2s = xs.transform(x2)
        xp = torch.from_numpy(x2s).float()
        yp=net(xp)
        yp = ys.inverse_transform(yp.detach().numpy())
        self.draw(x1,y1,x2,yp)
        print("end")

if __name__ == '__main__':
    n=NonLinearRegression()
    n.Train()
    print("ok")