import torch
import torch.nn as nn

from module import FSMNKernel


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fsmn = FSMNKernel(220, 0, 0, 1, 1)
        self.fc = nn.Linear(220, 3)

    def forward(self, x):
        return self.fc(nn.functional.relu(self.fsmn(x)))


if __name__ == '__main__':
    B, T, D = 32, 300, 220
    xx = torch.randn(B, T, D)
    yy = torch.randint(low=0, high=3, size=(32, 300)).view(-1)
    net = Net()
    optim = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.8)
    L = nn.CrossEntropyLoss()
    for _ in range(100):
        y = net(xx).view(-1, 3)
        loss = L(y, yy)
        print(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
