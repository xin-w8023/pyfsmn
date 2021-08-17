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
    lo, ro = 10, 10
    B, T, D = 10, 2000, 440
    num_iter = 10
    x = torch.arange(B*T*D).view(B, T, D).float()

    fsmnp = FSMNKernelParallel(D, lo, ro, padding_mode='zero')
    fsmnp.filter.weight.data = torch.arange(fsmnp.filter.weight.numel()).view(fsmnp.filter.weight.size()).float()
    fsmnp.filter.weight.data /= torch.max(fsmnp.filter.weight.data)

    s = time.time()
    fsmnp_out = x
    for _ in range(num_iter):
        fsmnp_out = fsmnp(fsmnp_out)
    fsmnp_time = time.time() - s

    fsmn = FSMNKernel(dims=D, l_order=lo, r_order=ro, l_stride=1, r_stride=1)
    fsmn.filter.data = nn.Parameter(torch.arange(fsmnp.filter.weight.numel()).view(D, lo+ro+1).transpose(0, 1).float())
    fsmn.filter.data /= torch.max(fsmn.filter.data)

    s = time.time()
    fsmn_out = x
    for _ in range(num_iter):
        fsmn_out = fsmn(fsmn_out)
    fsmn_time = time.time() - s
    print('#' * 80)
    print(f'maximum relative error: max(abs((fsmnp_out - fsmn_out)/ fsmnp_out)) ='
          f' {torch.max((torch.abs(fsmnp_out - fsmn_out) / (fsmnp_out + 1e-8))):.8f}')

    print('#' * 80)
    print(f'parallel fsmn kernel time used: {fsmnp_time:.8f}\n'
          f'for-loop fsmn kernel time used: {fsmn_time:.8f}')
    print('#' * 80)


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
