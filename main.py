import torch
import torch.nn as nn

from module import FSMNKernel, FSMNKernelParallel


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.fsmn = FSMNKernel(220, 0, 0, 1, 1)
        self.fsmn = FSMNKernelParallel(220, 1, 1)
        self.fc = nn.Linear(220, 3)

    def forward(self, x):
        return self.fc(nn.functional.relu(self.fsmn(x)))


def test():
    import time
    lo, ro = 2, 2
    B, T, D = 10, 100, 30
    num_iter = 1000
    x = torch.arange(B*T*D).view(B, T, D).float()

    fsmnp = FSMNKernelParallel(D, lo, ro, padding_mode='zero')
    fsmnp.filter.weight.data = torch.arange(fsmnp.filter.weight.numel()).view(fsmnp.filter.weight.size()).float()
    print('fsmnp filter:', fsmnp.filter.weight.data)
    s = time.time()
    for _ in range(num_iter):
        fsmnp_out = fsmnp(x)
    fsmnp_time = time.time() - s

    fsmn = FSMNKernel(dims=D, l_order=lo, r_order=ro, l_stride=1, r_stride=1)
    fsmn.filter.data = nn.Parameter(torch.arange(fsmnp.filter.weight.numel()).view(D, lo+ro+1).transpose(0, 1).float())
    print('fsmn filter:', fsmn.filter.data)
    s = time.time()
    for _ in range(num_iter):
        fsmn_out = fsmn(x)
    fsmn_time = time.time() - s

    print('#' * 40)
    print(f'diff: sum(fsmnp_out - fsmn_out) = {torch.sum(fsmnp_out - fsmn_out)}')

    print('#' * 40)
    print(f'parallel time used: {fsmnp_time}\n'
          f'for-loop time used: {fsmn_time}')
    print('#' * 40)


if __name__ == '__main__':
    # B, T, D = 32, 300, 220
    # xx = torch.randn(B, T, D)
    # yy = torch.randint(low=0, high=3, size=(32, 300)).view(-1)
    # net = Net()
    # optim = torch.optim.SGD(net.parameters(), lr=1e-1, momentum=0.8)
    # L = nn.CrossEntropyLoss()
    # for _ in range(100):
    #     y = net(xx).view(-1, 3)
    #     loss = L(y, yy)
    #     print(loss.item())
    #     optim.zero_grad()
    #     loss.backward()
    #     optim.step()
    test()
